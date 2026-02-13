import { useState } from "react";
import { JobFormData } from "@/types/job";
import { createJob } from "@/services/api";
import JobForm from "@/components/JobForm";
import JobStatusPanel from "@/components/JobStatusPanel";
import { toast } from "sonner";
import { Crosshair } from "lucide-react";

const Index = () => {
  const [jobId, setJobId] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [lastFormData, setLastFormData] = useState<Partial<JobFormData>>({});

  const handleSubmit = async (data: JobFormData) => {
    setIsSubmitting(true);
    setLastFormData(data);
    try {
      const result = await createJob(data);
      setJobId(result.id);
      toast.success("Job created successfully!");
    } catch (err: any) {
      toast.error(err.message || "Failed to start job");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRetry = () => {
    setJobId(null);
  };

  const handleNewJob = () => {
    setJobId(null);
    setLastFormData({});
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="border-b border-border">
        <div className="max-w-3xl mx-auto px-4 py-5 flex items-center gap-3">
          <Crosshair className="h-6 w-6 text-primary" />
          <div>
            <h1 className="text-lg font-bold font-mono text-foreground tracking-tight">
              YOLO Dataset Generator
            </h1>
            <p className="text-xs text-muted-foreground">
              Generate detection datasets from YouTube gameplay
            </p>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-3xl w-full mx-auto px-4 py-8 space-y-8">
        {!jobId ? (
          <section className="rounded-lg border border-border bg-card p-6">
            <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-5">
              Start new job
            </h2>
            <JobForm
              onSubmit={handleSubmit}
              initialData={lastFormData}
              isSubmitting={isSubmitting}
            />
          </section>
        ) : (
          <section className="rounded-lg border border-border bg-card p-6">
            <div className="flex items-center justify-between mb-5">
              <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
                Job status
              </h2>
              <button
                onClick={handleNewJob}
                className="text-xs text-primary hover:underline"
              >
                + New job
              </button>
            </div>
            <JobStatusPanel jobId={jobId} onRetry={handleRetry} />
          </section>
        )}
      </main>
    </div>
  );
};

export default Index;
