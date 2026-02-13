import { useEffect, useRef, useState } from "react";
import { JobStatus as JobStatusType } from "@/types/job";
import { getJobStatus } from "@/services/api";
import { Loader2, CheckCircle2, XCircle, Copy, Check, AlertTriangle } from "lucide-react";

interface Props {
  jobId: string;
  onRetry: () => void;
}

export default function JobStatusPanel({ jobId, onRetry }: Props) {
  const [job, setJob] = useState<JobStatusType | null>(null);
  const [pollError, setPollError] = useState(false);
  const [copied, setCopied] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let active = true;

    const poll = async () => {
      try {
        const data = await getJobStatus(jobId);
        if (!active) return;
        setJob(data);
        setPollError(false);
        if (data.status === "completed" || data.status === "failed") {
          if (intervalRef.current) clearInterval(intervalRef.current);
        }
      } catch {
        if (active) setPollError(true);
      }
    };

    poll();
    intervalRef.current = setInterval(poll, 5000);

    return () => {
      active = false;
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [jobId]);

  const copyPath = async (path: string) => {
    await navigator.clipboard.writeText(path);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const statusColor = {
    pending: "text-warning",
    running: "text-primary",
    completed: "text-accent",
    failed: "text-destructive",
  };

  const StatusIcon = () => {
    if (!job) return <Loader2 className="h-5 w-5 animate-spin-slow text-primary" />;
    switch (job.status) {
      case "pending":
      case "running":
        return <Loader2 className="h-5 w-5 animate-spin-slow text-primary" />;
      case "completed":
        return <CheckCircle2 className="h-5 w-5 text-accent" />;
      case "failed":
        return <XCircle className="h-5 w-5 text-destructive" />;
    }
  };

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center gap-3">
        <StatusIcon />
        <div className="min-w-0 flex-1">
          <p className="text-xs text-muted-foreground font-mono truncate">Job: {jobId}</p>
          <p className={`text-sm font-semibold capitalize ${job ? statusColor[job.status] : "text-muted-foreground"}`}>
            {job?.status || "Loading…"}
          </p>
        </div>
      </div>

      {pollError && (
        <div className="flex items-center gap-2 text-xs text-warning bg-warning/10 rounded-md px-3 py-2 border border-warning/20">
          <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
          Connection issue — retrying…
        </div>
      )}

      {job?.message && (
        <p className="text-sm text-muted-foreground">{job.message}</p>
      )}

      {/* Processing note */}
      {job && (job.status === "pending" || job.status === "running") && (
        <div className="rounded-md border border-border bg-muted/50 px-4 py-3 text-xs text-muted-foreground">
          ⏳ Processing can take <strong className="text-foreground">30–90 minutes</strong> on CPU. This is normal — sit tight!
        </div>
      )}

      {/* Summary */}
      {job?.status === "completed" && job.summary && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {[
              { label: "Frames processed", value: job.summary.frames_processed },
              { label: "Objects detected", value: job.summary.objects_detected },
              { label: "Errors", value: job.summary.errors },
              { label: "Train images", value: job.summary.train_images },
              { label: "Val images", value: job.summary.val_images },
            ].map((s) => (
              <div key={s.label} className="rounded-md border border-border bg-muted/40 px-3 py-2.5">
                <p className="text-[11px] text-muted-foreground uppercase tracking-wider">{s.label}</p>
                <p className="text-lg font-bold font-mono text-foreground">{s.value.toLocaleString()}</p>
              </div>
            ))}
          </div>

          {/* Classes */}
          <div>
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Detected classes</p>
            <div className="rounded-md border border-border overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border bg-muted/60">
                    <th className="text-left px-3 py-1.5 text-xs text-muted-foreground font-medium">ID</th>
                    <th className="text-left px-3 py-1.5 text-xs text-muted-foreground font-medium">Class name</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(job.summary.classes).map(([id, name]) => (
                    <tr key={id} className="border-b border-border last:border-0">
                      <td className="px-3 py-1.5 font-mono text-primary">{id}</td>
                      <td className="px-3 py-1.5 text-foreground">{name}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Dataset path */}
          <div className="space-y-1.5">
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Dataset path</p>
            <div className="flex items-center gap-2">
              <code className="flex-1 rounded-md border border-border bg-muted/40 px-3 py-2 text-xs font-mono text-foreground truncate">
                {job.summary.dataset_path}
              </code>
              <button
                onClick={() => copyPath(job.summary!.dataset_path)}
                className="shrink-0 rounded-md border border-border bg-secondary px-2.5 py-2 text-muted-foreground hover:text-foreground transition-colors"
                title="Copy path"
              >
                {copied ? <Check className="h-3.5 w-3.5 text-accent" /> : <Copy className="h-3.5 w-3.5" />}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Failed */}
      {job?.status === "failed" && (
        <div className="space-y-3">
          <div className="rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {job.message || "Job failed. Please try again."}
          </div>
          <button
            onClick={onRetry}
            className="rounded-md border border-border bg-secondary px-4 py-2 text-sm font-medium text-foreground hover:bg-muted transition-colors"
          >
            Try again
          </button>
        </div>
      )}
    </div>
  );
}
