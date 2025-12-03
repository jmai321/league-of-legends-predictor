import type { PredictionResponse, LiveGameState, DraftTeams } from '@/types';

export const mockPredictionResponse: PredictionResponse = {
  blue_win_prob: 0.62,
  red_win_prob: 0.38,
  factors: [
    { name: "Blue has Jinx", impact: 0.08 },
    { name: "Red has Thresh", impact: 0.05 },
    { name: "Gold difference", impact: 0.12 },
    { name: "Dragon control", impact: 0.07 },
  ]
};

export const mockLiveGameState: LiveGameState = {
  game_time: 15,
  gold_diff: 2500,
  dragons: "Mountain",
};

export const mockDraftTeams: DraftTeams = {
  red_team: ["Gnar", "Lee Sin", "Ahri", "Jinx", "Thresh"],
  blue_team: ["Irelia", "Elise", "Zed", "Ezreal", "Leona"]
};

export const mockApiCall = {
  predictLive: async (gameState: LiveGameState): Promise<PredictionResponse> => {
    await new Promise(resolve => setTimeout(resolve, 1000));
    return mockPredictionResponse;
  },
  
  predictDraft: async (teams: DraftTeams): Promise<PredictionResponse> => {
    await new Promise(resolve => setTimeout(resolve, 1000));
    return mockPredictionResponse;
  }
};