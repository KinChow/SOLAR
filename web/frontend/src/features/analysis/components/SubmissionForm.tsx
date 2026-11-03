import CodeMirror from "@uiw/react-codemirror";
import { python } from "@codemirror/lang-python";
import { Plus, Play, Trash2 } from "lucide-react";
import { FormEvent, useMemo, useState } from "react";

import type { HardwareOption, RunCreateRequest, TensorInput } from "../types";


const DEFAULT_CODE = `class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(128, 256))
        self.w2 = nn.Parameter(torch.randn(256, 128))
        self.w3 = nn.Parameter(torch.randn(128, 64))

    def forward(self, x):
        x = torch.matmul(x, self.w1)
        x = torch.matmul(x, self.w2)
        return torch.matmul(x, self.w3)`;


interface EditableTensor {
  name: string;
  shape: string;
  dtype: "float32";
}


interface Props {
  hardware: HardwareOption[];
  submitting: boolean;
  error: string | null;
  onSubmit: (request: RunCreateRequest) => Promise<void>;
}


export function SubmissionForm({ hardware, submitting, error, onSubmit }: Props) {
  const [code, setCode] = useState(DEFAULT_CODE);
  const [selectedHardware, setSelectedHardware] = useState("H100_PCIe");
  const [inputs, setInputs] = useState<EditableTensor[]>([
    { name: "x", shape: "32, 128", dtype: "float32" },
  ]);
  const [validationError, setValidationError] = useState<string | null>(null);
  const canAdd = inputs.length < 8;
  const effectiveHardware = useMemo(
    () => hardware.some((option) => option.id === selectedHardware) ? selectedHardware : hardware[0]?.id ?? "",
    [hardware, selectedHardware],
  );

  const updateInput = (index: number, patch: Partial<EditableTensor>) => {
    setInputs((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, ...patch } : item));
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    try {
      const normalized: TensorInput[] = inputs.map((input) => {
        const shape = input.shape.split(",").map((value) => Number(value.trim()));
        if (!input.name.match(/^[A-Za-z_][A-Za-z0-9_]*$/) || shape.some((value) => !Number.isInteger(value) || value <= 0)) {
          throw new Error("Input names and positive integer shapes are required");
        }
        return { name: input.name, shape, dtype: input.dtype };
      });
      setValidationError(null);
      await onSubmit({ operator_code: code, inputs: normalized, hardware: effectiveHardware });
    } catch (submitError) {
      setValidationError(submitError instanceof Error ? submitError.message : "Invalid input");
    }
  };

  return (
    <main className="submission-shell">
      <header className="page-heading">
        <div>
          <p className="eyebrow">Offline performance analysis</p>
          <h1>New analysis</h1>
        </div>
        <span className="trust-note">Trusted code only</span>
      </header>
      <form className="submission-grid" onSubmit={submit}>
        <section className="code-surface" aria-label="Operator implementation">
          <div className="surface-heading">
            <div><h2>Operator implementation</h2><span>Python · Model.forward</span></div>
          </div>
          <div>
            <CodeMirror aria-label="Operator code" value={code} height="350px" theme="dark" extensions={[python()]} onChange={setCode} />
          </div>
        </section>
        <aside className="input-surface">
          <div className="surface-heading"><div><h2>Input tensors</h2><span>{inputs.length} / 8</span></div></div>
          <div className="tensor-list">
            {inputs.map((input, index) => (
              <div className="tensor-row" key={index}>
                <label>Name<input aria-label={`Input name ${index + 1}`} value={input.name} onChange={(event) => updateInput(index, { name: event.target.value })} /></label>
                <label>Shape<input aria-label={index === 0 ? "Input shape" : `Input shape ${index + 1}`} value={input.shape} onChange={(event) => updateInput(index, { shape: event.target.value })} /></label>
                <label>Dtype<select aria-label={`Input dtype ${index + 1}`} value={input.dtype} disabled><option value="float32">float32</option></select></label>
                {inputs.length > 1 && <button className="icon-button danger" type="button" title="Remove tensor" onClick={() => setInputs((current) => current.filter((_, itemIndex) => itemIndex !== index))}><Trash2 size={16} /></button>}
              </div>
            ))}
          </div>
          <button className="text-action" type="button" disabled={!canAdd} onClick={() => setInputs((current) => [...current, { name: `x${current.length + 1}`, shape: "16, 16", dtype: "float32" }])}><Plus size={16} /> Add tensor</button>
          <label className="hardware-field">Target hardware
            <select value={effectiveHardware} onChange={(event) => setSelectedHardware(event.target.value)}>
              {hardware.map((option) => <option key={option.id} value={option.id}>{option.name}{option.validated ? " · validated" : " · unverified"}</option>)}
            </select>
          </label>
          {(validationError || error) && <div className="form-error" role="alert">{validationError || error}</div>}
          <button className="primary-button" type="submit" disabled={submitting || !effectiveHardware}><Play size={17} />{submitting ? "Starting…" : "Run analysis"}</button>
        </aside>
      </form>
    </main>
  );
}
