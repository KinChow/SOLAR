# Phase 1 Torch Capability Contract

Phase 1 is the PyTorch front-end and semantic-cost gate for the P800 offline
estimator. It does not promise that every API listed by XPyTorch can produce a
runtime estimate.

## Scope

The input is one or more concrete, already extracted `WorkloadGraph` cases.
Dynamic branches are represented as separate cases and are never silently
averaged. Raw Python execution is outside this contract. The target is a single
P800 device; multi-device collectives and M100 are out of scope.

The XPyTorch API list is an inventory source, not the estimator support set.
Aliases and Tensor/functional/Module entry points must map to one canonical
semantic family before cost rules are selected.

## Input contract

```yaml
contract_version: "phase1.v1"
model:
  name: Matmul
target:
  hardware: P800
  api_manifest: x-pytorch-v3.3.1.0
cases:
  - case_id: default
    workload: stage1/workload_graph.yaml
    guards: {}
options:
  dynamic_shape: false
  allow_opaque_single_node: true
```

Required input invariants:

- `source_sha256` is a 64-character SHA-256 digest.
- Every case has one extracted workload graph and a unique `case_id`.
- Dimensions are positive integers in Phase 1; symbolic dimensions are
  rejected with a structured failure during extraction.
- Phase 1 accepts only the dtypes and layouts listed by the selected support
  matrix. Unsupported combinations fail explicitly during normalization or
  cost analysis.
- The extracted graph must pass producer-consumer, tensor-id, alias and
  side-effect validation before semantic normalization starts.

## Support levels

Each API/node reports the highest completed level:

| Level | Meaning |
| --- | --- |
| `recognized` | API and arguments were parsed. |
| `normalized` | The API maps to a canonical semantic family. |
| `cost_complete` | All required Cost IR fields have verified rules. |
| `lb_supported` | The case has complete semantic cost and an evidence-backed P800 capability model sufficient to produce a defensible hard lower bound `T_lb`. |

`recognized` and `normalized` are not performance support. A graph containing
an unsupported node cannot produce a complete graph-level `T_lb`.

## Output contract

```yaml
contract_version: "phase1.v1"
status: complete | partial | failed
model_name: Matmul
target_hardware: P800
cases:
  - case_id: default
    status: complete | partial | failed
    nodes:
      - node_id: Model.matmul
        api: matmul
        semantic_family: tensor_contraction
        support_level: lb_supported
        reason: null
        source_location: /abs/path/model.py:12
    artifacts:
      workload_graph: stage1/workload_graph.yaml
      operator_ir: stage2/operator_ir.yaml
      execution_graph: stage2/execution_graph.yaml
      cost_ir: stage3/cost_ir.yaml
    performance:
      t_lb_us: 0.0
      assumptions: [perfect_resource_overlap, resident_inputs]
issues:
  - code: unsupported_operator
    stage: normalize
    node_id: Model.custom
    message: no registered semantic normalizer
    source_location: /abs/path/model.py:20
```

Output rules:

- `failed` means the requested case could not produce a valid workload graph.
- `partial` means a graph was extracted but at least one node is not
  `lb_supported`; `t_lb_us` must be `null` at graph level.
- `complete` means every node has a verified cost rule and a valid P800 lower
  bound.
- A single-node opaque API may return a node-level result, but an opaque node
  makes the containing graph `partial`.
- Every failure includes a stage, stable error code, source location when
  available, and a reproducible case id.

`T_lb_us` is a hard theoretical lower bound, not an engineering prediction. It
is derived from mandatory semantic work, unavoidable physical data movement,
and the target's physical capability limits. A faster valid observation
invalidates the bound or its evidence; the observed value must not be clamped.
Measured attainable throughput and empirical efficiency belong to a separate
prediction contract and must not be substituted for a physical limit.

## P800 capability evidence gate

P800 capability facts are admitted in this order:

1. Prefer versioned KernelWiki-Kunlun hardware specifications and programming
   model documents that state physical topology, widths, rates, capacities, or
   other architectural limits.
2. Derived limits must retain the source facts and the reproducible derivation,
   such as `port_count * frequency * bits_per_transfer`.
3. Use controlled microbenchmarks to validate or falsify a candidate capability
   model when the documentation is missing or ambiguous.
4. A measured best throughput proves only that the hardware can reach at least
   that rate. It does not by itself prove a physical maximum. If no defensible
   upper capability bound can be established, keep the affected resource
   unsupported and do not emit graph-level `T_lb_us`.

Shape-dependent attainable ceilings, profiler measurements, and empirical
efficiency tables may be attached as validation evidence, but they remain
separate from the hard capability values used by this contract.

## Phase 1 acceptance

1. The API inventory is fully classified into canonical semantic families.
2. P0 families have deterministic WorkloadGraph, Operator IR, Execution Graph
   and Cost IR mappings.
3. P0 cases produce `lb_supported` results with assumptions and evidence.
4. Unsupported API/shape/dtype/layout combinations are explicit and never
   receive a zero or fallback cost.
5. The same input, source hash, manifest version and hardware capability
   produces byte-equivalent artifacts.
