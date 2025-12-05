import type { DraftTeams, LineupRequest } from "@/src/types";

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