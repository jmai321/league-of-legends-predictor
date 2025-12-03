"use client";

import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Navigation } from "@/src/components/Navigation";
import { ChampionSelector } from "@/src/components/ChampionSelector";
import { ROLE_LABELS } from "@/src/constants/champions";

export default function DraftPredictionPage() {
  const [redTeam, setRedTeam] = useState<string[]>(Array(5).fill(""));
  const [blueTeam, setBlueTeam] = useState<string[]>(Array(5).fill(""));

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

  return (
    <div className="min-h-screen bg-background relative">
      <Navigation />
      
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-foreground">
            Draft Prediction Page
          </h1>
        </div>
        
        <Card className="max-w-6xl mx-auto p-6">
          <div className="grid md:grid-cols-2 gap-8">
            {/* Left side - Draft */}
            <div className="space-y-6">
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
                <Button className="cursor-pointer">Run</Button>
              </div>
            </div>
            
            {/* Right side - Analysis Details */}
            <div className="space-y-6">
              <h2 className="text-2xl font-semibold text-center">Analysis Details</h2>
              
              <Card className="p-6 min-h-96">
                <div className="space-y-4">
                  <div className="text-lg">
                    <span className="font-medium">Win Probability:</span> 80%
                  </div>
                  {Array.from({ length: 4 }, (_, i) => (
                    <div key={i} className="text-muted-foreground">...</div>
                  ))}
                </div>
              </Card>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}