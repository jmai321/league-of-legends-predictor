import type { DraftTeams, LineupRequest, TeamStats, RealtimeGame, PredictionModel } from "@/src/types";
import { getFieldsForModel } from "@/src/constants/gameStats";

export function transformDraftToLineup(teams: DraftTeams): LineupRequest {
  return {
    top_blue: teams.blue_team[0] || "",
    jng_blue: teams.blue_team[1] || "",
    mid_blue: teams.blue_team[2] || "",
    bot_blue: teams.blue_team[3] || "",
    sup_blue: teams.blue_team[4] || "",
    top_red: teams.red_team[0] || "",
    jng_red: teams.red_team[1] || "",
    mid_red: teams.red_team[2] || "",
    bot_red: teams.red_team[3] || "",
    sup_red: teams.red_team[4] || "",
  };
}

export function transformToRealtimeGame(
  blueStats: TeamStats,
  redStats: TeamStats,
  model: PredictionModel,
  gameId: string = `GAME_${Date.now()}`
): RealtimeGame {
  const fields = getFieldsForModel(model);
  
  const buildRow = (stats: TeamStats, side: "Blue" | "Red") => {
    const row: Record<string, any> = {
      gameid: gameId,
      side,
      teamname: stats.teamname || "",
    };
    
    for (const field of fields) {
      if (field.key !== "teamname") {
        const value = stats[field.key];
        row[field.key] = value !== undefined ? value : field.defaultValue;
      }
    }
    
    return row;
  };

  return {
    rows: [
      buildRow(blueStats, "Blue"),
      buildRow(redStats, "Red"),
    ],
  };
}