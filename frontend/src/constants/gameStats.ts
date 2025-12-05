import type { PredictionModel } from "@/src/types";

// Prediction Model Options
export const PREDICTION_MODELS = [
  { value: "10" as PredictionModel, label: "10 Minutes" },
  { value: "15" as PredictionModel, label: "15 Minutes" },
  { value: "20" as PredictionModel, label: "20 Minutes" },
  { value: "25" as PredictionModel, label: "25 Minutes" },
  { value: "full" as PredictionModel, label: "Full Game" },
] as const;

// Field Types
export type FieldType = "text" | "number" | "checkbox";

// Game Statistics Configuration
export interface GameStatConfig {
  key: string;
  label: string;
  type: FieldType;
  placeholder?: string;
  defaultValue: string | number | boolean;
  models: PredictionModel[];
}

// Base Fields (available in all models)
export const BASE_FIELDS: GameStatConfig[] = [
  {
    key: "teamname",
    label: "Team Name", 
    type: "text",
    placeholder: "Enter team name",
    defaultValue: "",
    models: ["10", "15", "20", "25", "full"]
  }
];

// Full Game Fields
export const FULL_GAME_FIELDS: GameStatConfig[] = [
  // Basic Stats
  { key: "kills", label: "Kills", type: "number", defaultValue: 15, models: ["full"] },
  { key: "deaths", label: "Deaths", type: "number", defaultValue: 8, models: ["full"] },
  { key: "assists", label: "Assists", type: "number", defaultValue: 25, models: ["full"] },
  { key: "teamkills", label: "Team Kills", type: "number", defaultValue: 15, models: ["full"] },
  { key: "teamdeaths", label: "Team Deaths", type: "number", defaultValue: 8, models: ["full"] },
  { key: "doublekills", label: "Double Kills", type: "number", defaultValue: 3, models: ["full"] },
  { key: "triplekills", label: "Triple Kills", type: "number", defaultValue: 1, models: ["full"] },
  { key: "quadrakills", label: "Quadra Kills", type: "number", defaultValue: 0, models: ["full"] },
  { key: "pentakills", label: "Penta Kills", type: "number", defaultValue: 0, models: ["full"] },

  // Objectives
  { key: "towers", label: "Towers", type: "number", defaultValue: 7, models: ["full"] },
  { key: "opp_towers", label: "Opp Towers", type: "number", defaultValue: 3, models: ["full"] },
  { key: "turretplates", label: "Turret Plates", type: "number", defaultValue: 8, models: ["full"] },
  { key: "opp_turretplates", label: "Opp Turret Plates", type: "number", defaultValue: 3, models: ["full"] },
  { key: "inhibitors", label: "Inhibitors", type: "number", defaultValue: 2, models: ["full"] },
  { key: "opp_inhibitors", label: "Opp Inhibitors", type: "number", defaultValue: 0, models: ["full"] },
  { key: "dragons", label: "Dragons", type: "number", defaultValue: 3, models: ["full"] },
  { key: "opp_dragons", label: "Opp Dragons", type: "number", defaultValue: 1, models: ["full"] },
  { key: "elementaldrakes", label: "Elemental Drakes", type: "number", defaultValue: 3, models: ["full"] },
  { key: "opp_elementaldrakes", label: "Opp Elemental Drakes", type: "number", defaultValue: 1, models: ["full"] },
  { key: "mountains", label: "Mountain Drakes", type: "number", defaultValue: 1, models: ["full"] },
  { key: "clouds", label: "Cloud Drakes", type: "number", defaultValue: 1, models: ["full"] },
  { key: "oceans", label: "Ocean Drakes", type: "number", defaultValue: 1, models: ["full"] },
  { key: "infernals", label: "Infernal Drakes", type: "number", defaultValue: 0, models: ["full"] },
  { key: "chemtechs", label: "Chemtech Drakes", type: "number", defaultValue: 0, models: ["full"] },
  { key: "hextechs", label: "Hextech Drakes", type: "number", defaultValue: 0, models: ["full"] },
  { key: "elders", label: "Elder Dragons", type: "number", defaultValue: 1, models: ["full"] },
  { key: "opp_elders", label: "Opp Elder Dragons", type: "number", defaultValue: 0, models: ["full"] },
  { key: "barons", label: "Barons", type: "number", defaultValue: 1, models: ["full"] },
  { key: "opp_barons", label: "Opp Barons", type: "number", defaultValue: 0, models: ["full"] },
  { key: "heralds", label: "Heralds", type: "number", defaultValue: 1, models: ["full"] },
  { key: "opp_heralds", label: "Opp Heralds", type: "number", defaultValue: 0, models: ["full"] },

  // Economy
  { key: "totalgold", label: "Total Gold", type: "number", defaultValue: 65000, models: ["full"] },
  { key: "earnedgold", label: "Earned Gold", type: "number", defaultValue: 50000, models: ["full"] },
  { key: "earned_gpm", label: "Earned GPM", type: "number", defaultValue: 1500, models: ["full"] },
  { key: "earnedgoldshare", label: "Earned Gold Share", type: "number", defaultValue: 0.2, models: ["full"] },
  { key: "goldspent", label: "Gold Spent", type: "number", defaultValue: 48000, models: ["full"] },
  { key: "gspd", label: "GSPD", type: "number", defaultValue: 1400, models: ["full"] },
  { key: "gpr", label: "GPR", type: "number", defaultValue: 1.05, models: ["full"] },

  // Combat
  { key: "damagetochampions", label: "Damage to Champions", type: "number", defaultValue: 25000, models: ["full"] },
  { key: "dpm", label: "DPM", type: "number", defaultValue: 750, models: ["full"] },
  { key: "damagetakenperminute", label: "Damage Taken per Min", type: "number", defaultValue: 600, models: ["full"] },
  { key: "damagemitigatedperminute", label: "Damage Mitigated per Min", type: "number", defaultValue: 400, models: ["full"] },
  { key: "damagetotowers", label: "Damage to Towers", type: "number", defaultValue: 8000, models: ["full"] },

  // Vision
  { key: "wardsplaced", label: "Wards Placed", type: "number", defaultValue: 25, models: ["full"] },
  { key: "wpm", label: "WPM", type: "number", defaultValue: 0.75, models: ["full"] },
  { key: "wardskilled", label: "Wards Killed", type: "number", defaultValue: 15, models: ["full"] },
  { key: "wcpm", label: "WCPM", type: "number", defaultValue: 0.45, models: ["full"] },
  { key: "controlwardsbought", label: "Control Wards Bought", type: "number", defaultValue: 8, models: ["full"] },
  { key: "visionscore", label: "Vision Score", type: "number", defaultValue: 45, models: ["full"] },
  { key: "vspm", label: "VSPM", type: "number", defaultValue: 1.35, models: ["full"] },

  // Farming
  { key: "minionkills", label: "Minion Kills", type: "number", defaultValue: 180, models: ["full"] },
  { key: "monsterkills", label: "Monster Kills", type: "number", defaultValue: 45, models: ["full"] },
  { key: "cspm", label: "CSPM", type: "number", defaultValue: 6.5, models: ["full"] },

  // Binary Fields
  { key: "firstblood", label: "First Blood", type: "checkbox", defaultValue: true, models: ["full"] },
  { key: "firstdragon", label: "First Dragon", type: "checkbox", defaultValue: true, models: ["full"] },
  { key: "firstbaron", label: "First Baron", type: "checkbox", defaultValue: false, models: ["full"] },
  { key: "firsttower", label: "First Tower", type: "checkbox", defaultValue: true, models: ["full"] },
  { key: "firstmidtower", label: "First Mid Tower", type: "checkbox", defaultValue: false, models: ["full"] },
  { key: "firsttothreetowers", label: "First to Three Towers", type: "checkbox", defaultValue: true, models: ["full"] },
  { key: "firstherald", label: "First Herald", type: "checkbox", defaultValue: true, models: ["full"] },
  { key: "team_kpm", label: "Team KPM", type: "number", defaultValue: 0.45, models: ["full"] },
  { key: "ckpm", label: "CKPM", type: "number", defaultValue: 0.35, models: ["full"] },
];

// Time-based fields generator
export const createTimeBasedFields = (minute: string): GameStatConfig[] => [
  { key: `goldat${minute}`, label: `Gold at ${minute}min`, type: "number", defaultValue: minute === "10" ? 18000 : minute === "15" ? 28000 : minute === "20" ? 38000 : 48000, models: [minute as PredictionModel] },
  { key: `xpat${minute}`, label: `XP at ${minute}min`, type: "number", defaultValue: minute === "10" ? 12000 : minute === "15" ? 19000 : minute === "20" ? 26000 : 33000, models: [minute as PredictionModel] },
  { key: `csat${minute}`, label: `CS at ${minute}min`, type: "number", defaultValue: minute === "10" ? 85 : minute === "15" ? 135 : minute === "20" ? 180 : 225, models: [minute as PredictionModel] },
  { key: `killsat${minute}`, label: `Kills at ${minute}min`, type: "number", defaultValue: minute === "10" ? 8 : minute === "15" ? 12 : minute === "20" ? 15 : 18, models: [minute as PredictionModel] },
  { key: `assistsat${minute}`, label: `Assists at ${minute}min`, type: "number", defaultValue: minute === "10" ? 12 : minute === "15" ? 18 : minute === "20" ? 24 : 30, models: [minute as PredictionModel] },
  { key: `deathsat${minute}`, label: `Deaths at ${minute}min`, type: "number", defaultValue: minute === "10" ? 3 : minute === "15" ? 6 : minute === "20" ? 8 : 10, models: [minute as PredictionModel] },
  { key: `opp_goldat${minute}`, label: `Opp Gold at ${minute}min`, type: "number", defaultValue: minute === "10" ? 17200 : minute === "15" ? 26500 : minute === "20" ? 36500 : 46500, models: [minute as PredictionModel] },
  { key: `opp_xpat${minute}`, label: `Opp XP at ${minute}min`, type: "number", defaultValue: minute === "10" ? 11500 : minute === "15" ? 18200 : minute === "20" ? 25200 : 32200, models: [minute as PredictionModel] },
  { key: `opp_csat${minute}`, label: `Opp CS at ${minute}min`, type: "number", defaultValue: minute === "10" ? 80 : minute === "15" ? 125 : minute === "20" ? 170 : 215, models: [minute as PredictionModel] },
  { key: `opp_killsat${minute}`, label: `Opp Kills at ${minute}min`, type: "number", defaultValue: minute === "10" ? 5 : minute === "15" ? 9 : minute === "20" ? 12 : 15, models: [minute as PredictionModel] },
  { key: `opp_assistsat${minute}`, label: `Opp Assists at ${minute}min`, type: "number", defaultValue: minute === "10" ? 8 : minute === "15" ? 14 : minute === "20" ? 20 : 26, models: [minute as PredictionModel] },
  { key: `opp_deathsat${minute}`, label: `Opp Deaths at ${minute}min`, type: "number", defaultValue: minute === "10" ? 8 : minute === "15" ? 12 : minute === "20" ? 15 : 18, models: [minute as PredictionModel] },
  { key: `golddiffat${minute}`, label: `Gold Diff at ${minute}min`, type: "number", defaultValue: 800, models: [minute as PredictionModel] },
  { key: `xpdiffat${minute}`, label: `XP Diff at ${minute}min`, type: "number", defaultValue: 500, models: [minute as PredictionModel] },
  { key: `csdiffat${minute}`, label: `CS Diff at ${minute}min`, type: "number", defaultValue: 15, models: [minute as PredictionModel] },
];

// All time-based fields
export const TIME_BASED_FIELDS: GameStatConfig[] = [
  ...createTimeBasedFields("10"),
  ...createTimeBasedFields("15"),
  ...createTimeBasedFields("20"),
  ...createTimeBasedFields("25"),
];

// All game statistics combined
export const ALL_GAME_STATS: GameStatConfig[] = [
  ...BASE_FIELDS,
  ...FULL_GAME_FIELDS,
  ...TIME_BASED_FIELDS,
];

// Helper function to get fields for a specific model
export const getFieldsForModel = (model: PredictionModel): GameStatConfig[] => {
  return ALL_GAME_STATS.filter(field => 
    field.models.includes(model)
  );
};