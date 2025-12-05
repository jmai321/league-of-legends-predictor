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