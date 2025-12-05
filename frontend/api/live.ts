import { apiRequest } from "./client";
import type { RealtimeGame, LiveResponse, PredictionModel } from "@/types";

export async function predictLive(
  request: RealtimeGame,
  model: PredictionModel
): Promise<LiveResponse> {
  const endpoint = model === "full" 
    ? "/predict/realtime/full"
    : `/predict/realtime/mid/${model}`;
  
  const result = await apiRequest<LiveResponse>(endpoint, {
    method: "POST",
    body: JSON.stringify(request),
  });

  // Add minute field for mid-game predictions if not present
  if (model !== "full" && !result.minute) {
    return { ...result, minute: parseInt(model) };
  }
  
  return result;
}