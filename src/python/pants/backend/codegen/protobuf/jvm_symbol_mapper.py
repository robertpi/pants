# Copyright 2023 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).
from __future__ import annotations

import os
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import DefaultDict

from pants.backend.codegen.protobuf.target_types import AllProtobufTargets, ProtobufSourceField
from pants.engine.addresses import Address
from pants.engine.fs import FileContent
from pants.engine.internals.graph import hydrate_sources
from pants.engine.intrinsics import get_digest_contents
from pants.engine.rules import collect_rules, concurrently, implicitly, rule
from pants.engine.target import HydrateSourcesRequest
from pants.jvm.dependency_inference.artifact_mapper import MutableTrieNode
from pants.jvm.dependency_inference.symbol_mapper import SymbolMap
from pants.jvm.subsystems import JvmSubsystem
from pants.jvm.target_types import JvmCodegenTypeField, JvmResolveField
from pants.util.ordered_set import OrderedSet

_ResolveName = str


@dataclass(frozen=True)
class FirstPartyProtobufJvmMappingRequest:
    # Which JVM language this mapping is being computed for -- "java" or "scala". Used both to
    # pick the generated symbol's expected capitalization (see `_determine_namespace`) and to
    # exclude targets whose `JvmCodegenTypeField` value names the other language, so that (once a
    # target is `parametrize`d by `jvm_codegen_type`) inferred dependencies point at the address
    # generating sources for the consumer's own language, rather than at an address that has been
    # configured to skip that language's codegen entirely.
    language: str


@rule
async def map_first_party_protobuf_jvm_targets_to_symbols(
    request: FirstPartyProtobufJvmMappingRequest,
    all_protobuf_targets: AllProtobufTargets,
    jvm: JvmSubsystem,
) -> SymbolMap:
    relevant_targets = tuple(
        tgt
        for tgt in all_protobuf_targets
        if (codegen_type := tgt.get(JvmCodegenTypeField).value) is None
        or codegen_type == request.language
    )

    sources = await concurrently(
        hydrate_sources(
            HydrateSourcesRequest(
                tgt[ProtobufSourceField],
                for_sources_types=(ProtobufSourceField,),
                enable_codegen=True,
            ),
            **implicitly(),
        )
        for tgt in relevant_targets
    )

    all_contents = await concurrently(
        get_digest_contents(source.snapshot.digest) for source in sources
    )

    namespace_mapping: DefaultDict[tuple[_ResolveName, str], OrderedSet[Address]] = defaultdict(
        OrderedSet
    )
    for tgt, contents in zip(relevant_targets, all_contents):
        if not contents:
            continue
        if len(contents) > 1:
            raise AssertionError(
                f"Protobuf target `{tgt.address}` mapped to more than one source file."
            )

        resolve = tgt[JvmResolveField].normalized_value(jvm)
        namespace = _determine_namespace(
            contents[0], capitalize_base_name=request.language == "java"
        )
        namespace_mapping[(resolve, namespace)].add(tgt.address)

    mapping: Mapping[str, MutableTrieNode] = defaultdict(MutableTrieNode)
    for (resolve, namespace), addresses in namespace_mapping.items():
        mapping[resolve].insert(namespace, addresses, first_party=True, recursive=True)

    return SymbolMap((resolve, node.frozen()) for resolve, node in mapping.items())


# Determine generated Java/Scala package name
# * https://grpc.io/docs/languages/java/generated-code
# * https://scalapb.github.io/docs/generated-code
def _determine_namespace(file: FileContent, *, capitalize_base_name: bool) -> str:
    base_name, _, _ = os.path.basename(file.path).partition(".")
    base_name = base_name.capitalize() if capitalize_base_name else base_name
    package_definition = _parse_package_definition(file.content)
    return f"{package_definition}.{base_name}" if package_definition else base_name


_QUOTE_CHAR = r"(?:'|\")"
_JAVA_PACKAGE_OPTION_RE = re.compile(
    rf"^\s*option\s+java_package\s+=\s+{_QUOTE_CHAR}(.+){_QUOTE_CHAR};"
)
_PACKAGE_RE = re.compile(r"^\s*package\s+(.+);")


def _parse_package_definition(content_raw: bytes) -> str | None:
    content = content_raw.decode()
    for line in content.splitlines():
        m = _JAVA_PACKAGE_OPTION_RE.match(line)
        if m:
            return m.group(1)
        m = _PACKAGE_RE.match(line)
        if m:
            return m.group(1)
    return None


def rules():
    return collect_rules()
