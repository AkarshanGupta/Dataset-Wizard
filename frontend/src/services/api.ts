import { JobFormData, JobStatus } from "@/types/job";

const BASE_URL = "http://localhost:10000";

export async function createJob(data: JobFormData): Promise<JobStatus> {
  // Transform frontend field names to backend field names
  const payload = {
    url: data.url,
    game_name: data.game,
    max_frames: data.frames,
    model_choice: data.model,
    frame_rate: data.frame_rate,
    quick_test: data.quick_test,
  };
  
  const res = await fetch(`${BASE_URL}/api/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(err?.detail || `Server error: ${res.status}`);
  }
  return res.json();
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${BASE_URL}/api/jobs/${jobId}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch job status: ${res.status}`);
  }
  return res.json();
}
