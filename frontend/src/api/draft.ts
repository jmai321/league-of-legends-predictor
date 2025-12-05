import { apiRequest } from "./client";
import type { LineupRequest, LineupResponse } from "@/src/types";

export async function predictLineup(request: LineupRequest): Promise<LineupResponse> {
  return apiRequest<LineupResponse>("/predict/lineup", {
    method: "POST",
    body: JSON.stringify(request),
  });
}