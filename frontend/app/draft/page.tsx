"use client";

import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { PageLayout } from "@/components/layout/PageLayout";
import { ChampionSelector } from "@/components/ChampionSelector";
import { ROLE_LABELS } from "@/constants/champions";
import { predictLineup, ApiError } from "@/api";
import { transformDraftToLineup } from "@/lib/transformers";
import type { DraftTeams, LineupResponse } from "@/types";

export default function DraftPredictionPage() {
  const [redTeam, setRedTeam] = useState<string[]>(["K'Sante", "Maokai", "Tristana", "Jinx", "Lulu"]);
  const [blueTeam, setBlueTeam] = useState<string[]>(["Sion", "Lee Sin", "Ahri", "Aphelios", "Braum"]);
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<LineupResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const updateRedChampion = (index: number, champion: string) => {
    const newRedTeam = [...redTeam];
    newRedTeam[index] = champion;
    setRedTeam(newRedTeam);
  };

  const updateBlueChampion = (index: number, champion: string) => {
    const newBlueTeam = [...blueTeam];
    newBlueTeam[index] = champion;
    setBlueTeam(newBlueTeam);
  };

  const handleRunPrediction = async () => {
    const teams: DraftTeams = { red_team: redTeam, blue_team: blueTeam };
    
    if (teams.red_team.some(champ => !champ) || teams.blue_team.some(champ => !champ)) {
      setError("Please select all champions before running prediction");
      return;
    }

    setIsLoading(true);
    setError(null);
    
    try {
      const request = transformDraftToLineup(teams);
      const response = await predictLineup(request);
      setPrediction(response);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(`API Error: ${err.message}`);
      } else {
        setError("An unexpected error occurred");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <PageLayout title="Draft Prediction">
      <Card className="max-w-6xl mx-auto p-6">
          <div className="grid md:grid-cols-2 gap-8">
            {/* Left side - Draft */}
            <div className="space-y-6 h-full">
              <h2 className="text-2xl font-semibold text-center">Draft</h2>
              
              <div className="grid grid-cols-2 gap-4">
                {/* Red Team */}
                <div className="space-y-4 flex flex-col items-center">
                  <h3 className="text-lg font-medium text-red-600 text-center">Red</h3>
                  {Array.from({ length: 5 }, (_, i) => (
                    <ChampionSelector
                      key={`red-${i}`}
                      value={redTeam[i]}
                      onChange={(champion) => updateRedChampion(i, champion)}
                      placeholder={ROLE_LABELS[i]}
                    />
                  ))}
                </div>
                
                {/* Blue Team */}
                <div className="space-y-4 flex flex-col items-center">
                  <h3 className="text-lg font-medium text-blue-600 text-center">Blue</h3>
                  {Array.from({ length: 5 }, (_, i) => (
                    <ChampionSelector
                      key={`blue-${i}`}
                      value={blueTeam[i]}
                      onChange={(champion) => updateBlueChampion(i, champion)}
                      placeholder={ROLE_LABELS[i]}
                    />
                  ))}
                </div>
              </div>
              
              <div className="text-center">
                <Button 
                  className="cursor-pointer"
                  onClick={handleRunPrediction}
                  disabled={isLoading}
                >
                  {isLoading ? "Running..." : "Run"}
                </Button>
              </div>
            </div>
            
            {/* Right side - Analysis Details */}
            <div className="space-y-6 h-full">
              <h2 className="text-2xl font-semibold text-center">Analysis Details</h2>
              
              <Card className="p-6 h-96 overflow-y-auto">
                {error && (
                  <div className="text-red-600 mb-4 p-3 border border-red-200 rounded">
                    {error}
                  </div>
                )}
                
                {prediction ? (
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <div className="text-lg">
                        <span className="font-medium">Blue Win Probability:</span> {(prediction.p_blue * 100).toFixed(1)}%
                      </div>
                      <div className="text-lg">
                        <span className="font-medium">Red Win Probability:</span> {(prediction.p_red * 100).toFixed(1)}%
                      </div>
                    </div>
                    
                    <div className="mt-6">
                      <h3 className="font-medium mb-1">Top Features:</h3>
                      <div className="space-y-1 text-sm">
                        {prediction.top_features.slice(0, 10).map((feature, i) => (
                          <div key={i} className="flex justify-between">
                            <span>{feature.feature}</span>
                            <span>{feature.importance.toFixed(3)}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="mt-4">
                      <h3 className="font-medium mb-1">Feature Contributions:</h3>
                      <div className="space-y-1 text-sm">
                        {prediction.feature_contribs.slice(0, 10).map((contrib, i) => (
                          <div key={i} className="flex justify-between">
                            <span>{contrib.feature}</span>
                            <span>{contrib.contribution.toFixed(3)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-muted-foreground text-center h-full flex items-center justify-center">
                    Select champions and run prediction to see analysis
                  </div>
                )}
              </Card>
            </div>
          </div>
      </Card>
    </PageLayout>
  );
}