// Backend API Request Types
export interface LineupRequest {
  bot_blue: string;
  jng_blue: string;
  mid_blue: string;
  sup_blue: string;
  top_blue: string;
  bot_red: string;
  jng_red: string;
  mid_red: string;
  sup_red: string;
  top_red: string;
}

// Backend API Response Types
export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface FeatureContribution {
  feature: string;
  contribution: number;
}

export interface LineupResponse {
  p_blue: number;
  p_red: number;
  top_features: FeatureImportance[];
  feature_contribs: FeatureContribution[];
}

// Frontend UI Types
export interface DraftTeams {
  red_team: string[];
  blue_team: string[];
}

// Live Prediction Types
export interface RealtimeRow {
  gameid: string;
  side: "Blue" | "Red";
  teamname?: string;
  [key: string]: string | number | undefined;
}

export interface RealtimeGame {
  rows: [RealtimeRow, RealtimeRow];
}

export interface LiveResponse {
  gameid: string;
  blue_team: string;
  red_team: string;
  p_blue_raw: number;
  p_red_raw: number;
  p_blue_norm: number;
  p_red_norm: number;
  top_features: FeatureImportance[];
  feature_contribs_blue: FeatureContribution[];
  feature_contribs_red: FeatureContribution[];
  minute?: number;
}

export type PredictionModel = "full" | "10" | "15" | "20" | "25";

export interface TeamStats {
  [key: string]: string | number | boolean | undefined;
}