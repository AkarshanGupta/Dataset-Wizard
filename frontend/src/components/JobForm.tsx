import { useState } from "react";
import { JobFormData } from "@/types/job";
import { Loader2 } from "lucide-react";

interface JobFormProps {
  onSubmit: (data: JobFormData) => Promise<void>;
  initialData?: Partial<JobFormData>;
  isSubmitting: boolean;
}

const DEFAULT_DATA: JobFormData = {
  url: "",
  game: "subway_surfers",
  frames: 200,
  model: "n",
  frame_rate: 60,
  quick_test: false,
  resume: false,
};

export default function JobForm({ onSubmit, initialData, isSubmitting }: JobFormProps) {
  const [form, setForm] = useState<JobFormData>({ ...DEFAULT_DATA, ...initialData });
  const [errors, setErrors] = useState<Record<string, string>>({});

  const validate = (): boolean => {
    const errs: Record<string, string> = {};
    if (!form.url.trim()) errs.url = "YouTube URL is required";
    else if (!/^https?:\/\/.+/.test(form.url.trim())) errs.url = "Enter a valid URL";
    if (!form.game.trim()) errs.game = "Game name is required";
    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validate()) onSubmit(form);
  };

  const update = <K extends keyof JobFormData>(key: K, value: JobFormData[K]) => {
    setForm((f) => ({ ...f, [key]: value }));
    if (errors[key]) setErrors((e) => ({ ...e, [key]: "" }));
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      {/* URL */}
      <div className="space-y-1.5">
        <label className="text-sm font-medium text-foreground">YouTube URL <span className="text-destructive">*</span></label>
        <input
          type="text"
          value={form.url}
          onChange={(e) => update("url", e.target.value)}
          placeholder="https://youtube.com/watch?v=..."
          className="w-full rounded-md border border-border bg-input px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring font-mono"
        />
        {errors.url && <p className="text-xs text-destructive">{errors.url}</p>}
      </div>

      {/* Game + Model row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="space-y-1.5">
          <label className="text-sm font-medium text-foreground">Game name <span className="text-destructive">*</span></label>
          <input
            type="text"
            value={form.game}
            onChange={(e) => update("game", e.target.value)}
            className="w-full rounded-md border border-border bg-input px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring font-mono"
          />
          {errors.game && <p className="text-xs text-destructive">{errors.game}</p>}
        </div>
        <div className="space-y-1.5">
          <label className="text-sm font-medium text-foreground">YOLO Model Size</label>
          <select
            value={form.model}
            onChange={(e) => update("model", e.target.value)}
            className="w-full rounded-md border border-border bg-input px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
          >
            <option value="n">Nano (fastest)</option>
            <option value="s">Small</option>
            <option value="m">Medium</option>
            <option value="l">Large</option>
            <option value="x">XLarge (best accuracy)</option>
          </select>
        </div>
      </div>

      {/* Frames + Frame rate row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="space-y-1.5">
          <label className="text-sm font-medium text-foreground">Frames</label>
          <input
            type="number"
            min={1}
            value={form.frames}
            onChange={(e) => update("frames", Math.max(1, parseInt(e.target.value) || 1))}
            className="w-full rounded-md border border-border bg-input px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring font-mono"
          />
        </div>
        <div className="space-y-1.5">
          <label className="text-sm font-medium text-foreground">Frame rate (every Nth frame)</label>
          <input
            type="number"
            min={1}
            value={form.frame_rate}
            onChange={(e) => update("frame_rate", Math.max(1, parseInt(e.target.value) || 1))}
            className="w-full rounded-md border border-border bg-input px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring font-mono"
          />
        </div>
      </div>

      {/* Checkboxes */}
      <div className="flex flex-wrap gap-6">
        <label className="flex items-center gap-2 text-sm text-foreground cursor-pointer">
          <input
            type="checkbox"
            checked={form.quick_test}
            onChange={(e) => update("quick_test", e.target.checked)}
            className="rounded border-border bg-input accent-primary h-4 w-4"
          />
          Quick test (~10 frames)
        </label>
        <label className="flex items-center gap-2 text-sm text-foreground cursor-pointer">
          <input
            type="checkbox"
            checked={form.resume}
            onChange={(e) => update("resume", e.target.checked)}
            className="rounded border-border bg-input accent-primary h-4 w-4"
          />
          Resume from checkpoint
        </label>
      </div>

      <button
        type="submit"
        disabled={isSubmitting}
        className="w-full flex items-center justify-center gap-2 rounded-md bg-primary px-4 py-2.5 text-sm font-semibold text-primary-foreground hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed glow-primary"
      >
        {isSubmitting ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin-slow" />
            Starting…
          </>
        ) : (
          "Start Job"
        )}
      </button>
    </form>
  );
}
