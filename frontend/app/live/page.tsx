"use client";

import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Navigation } from "@/src/components/Navigation";
import { TeamStatsForm } from "@/src/components/TeamStatsForm";
import { ModelSelector } from "@/src/components/ModelSelector";
import { predictLive, ApiError } from "@/src/api";
import { transformToRealtimeGame } from "@/src/lib/transformers";
import type { TeamStats, PredictionModel, LiveResponse } from "@/src/types";

export default function LivePredictionPage() {
  const [model, setModel] = useState<PredictionModel>("15");
  const [blueStats, setBlueStats] = useState<TeamStats>({ teamname: "" });
  const [redStats, setRedStats] = useState<TeamStats>({ teamname: "" });
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<LiveResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>("statistics");

  const handleRunPrediction = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const realtimeGame = transformToRealtimeGame(blueStats, redStats, model);
      const result = await predictLive(realtimeGame, model);
      setPrediction(result);
      setActiveTab("results");
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

  const isFormValid = () => {
    return blueStats.teamname.trim() !== "" && redStats.teamname.trim() !== "";
  };

  const renderResults = () => {
    if (error) {
      return (
        <div className="text-red-600 text-center p-3 border border-red-200 rounded">
          {error}
        </div>
      );
    }

    if (!prediction) {
      return (
        <div className="text-muted-foreground text-center h-96 flex items-center justify-center">
          Run prediction from Team Statistics tab to see analysis
        </div>
      );
    }

    return (
      <div className="space-y-6">
        {/* Win Probabilities */}
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-blue-600 font-medium">🔵 {prediction.blue_team}</span>
            <span className="font-semibold">{(prediction.p_blue_norm * 100).toFixed(1)}%</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-red-600 font-medium">🔴 {prediction.red_team}</span>
            <span className="font-semibold">{(prediction.p_red_norm * 100).toFixed(1)}%</span>
          </div>
          {prediction.minute && (
            <div className="text-sm text-muted-foreground text-center">
              Prediction at {prediction.minute} minutes
            </div>
          )}
        </div>

        {/* Features Analysis */}
        <Tabs defaultValue="top-features" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="top-features">Top Features</TabsTrigger>
            <TabsTrigger value="blue-impact">Blue Impact</TabsTrigger>
            <TabsTrigger value="red-impact">Red Impact</TabsTrigger>
          </TabsList>

          <TabsContent value="top-features" className="mt-4">
            <div className="h-48 overflow-y-auto overflow-y-scroll" style={{ scrollbarWidth: 'thin' }}>
              <div className="space-y-1 text-sm pr-2">
                {prediction.top_features.map((feature, i) => (
                  <div key={i} className="flex justify-between">
                    <span>{feature.feature}</span>
                    <span className="text-muted-foreground">{feature.importance.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="blue-impact" className="mt-4">
            <div className="h-48 overflow-y-auto overflow-y-scroll" style={{ scrollbarWidth: 'thin' }}>
              <div className="space-y-1 text-sm pr-2">
                {prediction.feature_contribs_blue.map((contrib, i) => (
                  <div key={i} className="flex justify-between">
                    <span>{contrib.feature}</span>
                    <span className={contrib.contribution >= 0 ? "text-green-600" : "text-red-600"}>
                      {contrib.contribution >= 0 ? "+" : ""}{contrib.contribution.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="red-impact" className="mt-4">
            <div className="h-48 overflow-y-auto overflow-y-scroll" style={{ scrollbarWidth: 'thin' }}>
              <div className="space-y-1 text-sm pr-2">
                {prediction.feature_contribs_red.map((contrib, i) => (
                  <div key={i} className="flex justify-between">
                    <span>{contrib.feature}</span>
                    <span className={contrib.contribution >= 0 ? "text-green-600" : "text-red-600"}>
                      {contrib.contribution >= 0 ? "+" : ""}{contrib.contribution.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-background relative">
      <Navigation />
      
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-foreground">
            Live Game Prediction
          </h1>
        </div>
        
        <Card className="max-w-6xl mx-auto p-6">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="statistics">Team Statistics</TabsTrigger>
              <TabsTrigger value="results">Analysis Results</TabsTrigger>
            </TabsList>

            <TabsContent value="statistics" className="mt-4">
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <h2 className="text-2xl font-semibold">Team Statistics</h2>
                  <div className="w-48">
                    <ModelSelector value={model} onChange={setModel} />
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <TeamStatsForm
                    side="Blue"
                    stats={blueStats}
                    model={model}
                    onChange={setBlueStats}
                  />
                  
                  <TeamStatsForm
                    side="Red"
                    stats={redStats}
                    model={model}
                    onChange={setRedStats}
                  />
                </div>
                
                <div className="text-center">
                  <Button 
                    className="cursor-pointer"
                    onClick={handleRunPrediction}
                    disabled={isLoading || !isFormValid()}
                  >
                    {isLoading ? "Running..." : "Run"}
                  </Button>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="results" className="mt-6">
              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-center">Analysis Results</h2>
                {renderResults()}
              </div>
            </TabsContent>
          </Tabs>
        </Card>
      </div>
    </div>
  );
}