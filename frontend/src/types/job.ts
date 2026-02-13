export interface Summary {
  frames_processed: number;
  objects_detected: number;
  errors: number;
  train_images: number;
  val_images: number;
  classes: Record<string, string>;
  dataset_path: string;
}

export interface JobStatus {
  id: string;
  status: "pending" | "running" | "completed" | "failed";
  message?: string | null;
  summary?: Summary | null;
}

export interface JobFormData {
  url: string;
  game: string;
  frames: number;
  model: "n" | "s" | "m" | "l" | "x"; // YOLO model sizes
  frame_rate: number;
  quick_test: boolean;
  resume: boolean;
}
