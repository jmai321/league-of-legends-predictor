export interface Champion {
  id: string;
  name: string;
}

export interface PredictionResponse {
  blue_win_prob: number;
  red_win_prob: number;
  factors: FeatureFactor[];
}

export interface FeatureFactor {
  name: string;
  impact: number;
}

export interface LiveGameState {
  game_time: number;
  gold_diff: number;
  dragons: string;
  [key: string]: string | number;
}

export interface DraftTeams {
  red_team: string[];
  blue_team: string[];
}