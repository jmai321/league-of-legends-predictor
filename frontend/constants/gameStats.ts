import type { PredictionModel } from "@/types";

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
  // Basic Stats (Blue=winning team, Red=losing team defaults)
  { key: "kills", label: "Kills", type: "number", defaultValue: 18, models: ["full"] },
  { key: "deaths", label: "Deaths", type: "number", defaultValue: 10, models: ["full"] },
  { key: "assists", label: "Assists", type: "number", defaultValue: 35, models: ["full"] },
  { key: "teamkills", label: "Team Kills", type: "number", defaultValue: 18, models: ["full"] },
  { key: "teamdeaths", label: "Team Deaths", type: "number", defaultValue: 10, models: ["full"] },
  { key: "doublekills", label: "Double Kills", type: "number", defaultValue: 2, models: ["full"] },
  { key: "triplekills", label: "Triple Kills", type: "number", defaultValue: 1, models: ["full"] },
  { key: "quadrakills", label: "Quadra Kills", type: "number", defaultValue: 0, models: ["full"] },
  { key: "pentakills", label: "Penta Kills", type: "number", defaultValue: 0, models: ["full"] },

  // Objectives (realistic winning team stats)
  { key: "towers", label: "Towers", type: "number", defaultValue: 8, models: ["full"] },
  { key: "turretplates", label: "Turret Plates", type: "number", defaultValue: 4, models: ["full"] },
  { key: "inhibitors", label: "Inhibitors", type: "number", defaultValue: 2, models: ["full"] },
  { key: "dragons", label: "Dragons", type: "number", defaultValue: 3, models: ["full"] },
  { key: "elementaldrakes", label: "Elemental Drakes", type: "number", defaultValue: 3, models: ["full"] },
  { key: "mountains", label: "Mountain Drakes", type: "number", defaultValue: 1, models: ["full"] },
  { key: "clouds", label: "Cloud Drakes", type: "number", defaultValue: 1, models: ["full"] },
  { key: "oceans", label: "Ocean Drakes", type: "number", defaultValue: 1, models: ["full"] },
  { key: "infernals", label: "Infernal Drakes", type: "number", defaultValue: 0, models: ["full"] },
  { key: "chemtechs", label: "Chemtech Drakes", type: "number", defaultValue: 0, models: ["full"] },
  { key: "hextechs", label: "Hextech Drakes", type: "number", defaultValue: 0, models: ["full"] },
  { key: "elders", label: "Elder Dragons", type: "number", defaultValue: 0, models: ["full"] },
  { key: "barons", label: "Barons", type: "number", defaultValue: 1, models: ["full"] },
  { key: "heralds", label: "Heralds", type: "number", defaultValue: 1, models: ["full"] },

  // Economy (realistic 30min winning game)
  { key: "totalgold", label: "Total Gold", type: "number", defaultValue: 60000, models: ["full"] },
  { key: "earnedgold", label: "Earned Gold", type: "number", defaultValue: 45000, models: ["full"] },
  { key: "goldspent", label: "Gold Spent", type: "number", defaultValue: 43000, models: ["full"] },

  // Combat
  { key: "damagetochampions", label: "Damage to Champions", type: "number", defaultValue: 20000, models: ["full"] },
  { key: "damagetotowers", label: "Damage to Towers", type: "number", defaultValue: 5000, models: ["full"] },

  // Vision
  { key: "wardsplaced", label: "Wards Placed", type: "number", defaultValue: 30, models: ["full"] },
  { key: "wardskilled", label: "Wards Killed", type: "number", defaultValue: 20, models: ["full"] },
  { key: "controlwardsbought", label: "Control Wards Bought", type: "number", defaultValue: 8, models: ["full"] },
  { key: "visionscore", label: "Vision Score", type: "number", defaultValue: 50, models: ["full"] },

  // Farming
  { key: "minionkills", label: "Minion Kills", type: "number", defaultValue: 200, models: ["full"] },
  { key: "monsterkills", label: "Monster Kills", type: "number", defaultValue: 50, models: ["full"] },

  // Binary Fields (Blue team defaults to getting first objectives)
  { key: "firstblood", label: "First Blood", type: "checkbox", defaultValue: true, models: ["full"] },
  { key: "firstdragon", label: "First Dragon", type: "checkbox", defaultValue: true, models: ["full"] },
  { key: "firstbaron", label: "First Baron", type: "checkbox", defaultValue: true, models: ["full"] },
  { key: "firsttower", label: "First Tower", type: "checkbox", defaultValue: true, models: ["full"] },
  { key: "firstmidtower", label: "First Mid Tower", type: "checkbox", defaultValue: true, models: ["full"] },
  { key: "firsttothreetowers", label: "First to Three Towers", type: "checkbox", defaultValue: true, models: ["full"] },
  { key: "firstherald", label: "First Herald", type: "checkbox", defaultValue: true, models: ["full"] },
];

// Time-based fields generator
export const createTimeBasedFields = (minute: string): GameStatConfig[] => [
  { key: `goldat${minute}`, label: `Gold at ${minute}min`, type: "number", defaultValue: minute === "10" ? 3400 : minute === "15" ? 5500 : minute === "20" ? 7600 : 9800, models: [minute as PredictionModel] },
  { key: `xpat${minute}`, label: `XP at ${minute}min`, type: "number", defaultValue: minute === "10" ? 4800 : minute === "15" ? 7800 : minute === "20" ? 10500 : 13000, models: [minute as PredictionModel] },
  { key: `csat${minute}`, label: `CS at ${minute}min`, type: "number", defaultValue: minute === "10" ? 85 : minute === "15" ? 140 : minute === "20" ? 185 : 230, models: [minute as PredictionModel] },
  { key: `killsat${minute}`, label: `Kills at ${minute}min`, type: "number", defaultValue: minute === "10" ? 2 : minute === "15" ? 4 : minute === "20" ? 6 : 8, models: [minute as PredictionModel] },
  { key: `assistsat${minute}`, label: `Assists at ${minute}min`, type: "number", defaultValue: minute === "10" ? 3 : minute === "15" ? 6 : minute === "20" ? 10 : 15, models: [minute as PredictionModel] },
  { key: `deathsat${minute}`, label: `Deaths at ${minute}min`, type: "number", defaultValue: minute === "10" ? 1 : minute === "15" ? 2 : minute === "20" ? 3 : 4, models: [minute as PredictionModel] },
  { key: `golddiffat${minute}`, label: `Gold Diff at ${minute}min`, type: "number", defaultValue: minute === "10" ? 200 : minute === "15" ? 400 : minute === "20" ? 600 : 800, models: [minute as PredictionModel] },
  { key: `xpdiffat${minute}`, label: `XP Diff at ${minute}min`, type: "number", defaultValue: minute === "10" ? 150 : minute === "15" ? 300 : minute === "20" ? 450 : 600, models: [minute as PredictionModel] },
  { key: `csdiffat${minute}`, label: `CS Diff at ${minute}min`, type: "number", defaultValue: minute === "10" ? 5 : minute === "15" ? 10 : minute === "20" ? 15 : 20, models: [minute as PredictionModel] },
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

// Helper function to get appropriate default values based on team side
export const getDefaultValue = (field: GameStatConfig, side: "Blue" | "Red"): string | number | boolean => {
  // For binary "first" objectives
  if (field.type === "checkbox" && field.key.startsWith("first")) {
    switch (field.key) {
      case "firstblood": return side === "Red";
      case "firstdragon": return side === "Red";
      case "firstherald": return side === "Blue";
      case "firstbaron": return side === "Blue";
      case "firsttower": return side === "Blue";
      case "firstmidtower": return side === "Red";
      case "firsttothreetowers": return side === "Red";
      default: return side === "Blue";
    }
  }
  
  if (field.type === "number" && side === "Blue") {
    switch (field.key) {
      case "kills": case "teamkills": return 19;
      case "deaths": case "teamdeaths": return 14;
      case "assists": return 62;
      case "towers": return 9;
      case "dragons": return 2;
      case "elementaldrakes": return 2; 
      case "heralds": return 1;
      case "barons": return 2;
      case "totalgold": return 66215;
      case "earnedgold": return 43509;
      case "goldspent": return 60375;
      case "goldat10": return 16076;
      case "goldat15": return 24312; 
      case "goldat20": return 33674;
      case "golddiffat10": return -938;
      case "golddiffat15": return -2204;
      case "golddiffat20": return -1532;
    }
  }
  
  if (field.type === "number" && side === "Red") {
    switch (field.key) {
      case "kills": case "teamkills": return 14;
      case "deaths": case "teamdeaths": return 19;
      case "assists": return 26;
      case "towers": return 4;
      case "dragons": return 3;
      case "elementaldrakes": return 3;
      case "heralds": return 1;
      case "barons": return 0;
      case "totalgold": return 60011;
      case "earnedgold": return 37305;
      case "goldspent": return 58275;
      case "goldat10": return 17014;
      case "goldat15": return 26516;
      case "goldat20": return 35206;
      case "golddiffat10": return 938;
      case "golddiffat15": return 2204;
      case "golddiffat20": return 1532;
    }
  }
  
  return field.defaultValue;
};