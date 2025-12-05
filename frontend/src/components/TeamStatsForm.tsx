"use client";

import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { getFieldsForModel } from "@/src/constants/gameStats";
import type { TeamStats, PredictionModel } from "@/src/types";

interface TeamStatsFormProps {
  side: "Blue" | "Red";
  stats: TeamStats;
  model: PredictionModel;
  onChange: (stats: TeamStats) => void;
}

export function TeamStatsForm({ side, stats, model, onChange }: TeamStatsFormProps) {
  const fields = getFieldsForModel(model);
  const sideColor = side === "Blue" ? "text-blue-600" : "text-red-600";

  const handleInputChange = (key: string, value: string | number | boolean) => {
    const newStats = { ...stats };
    newStats[key as keyof TeamStats] = value as any;
    onChange(newStats);
  };

  return (
    <Card className="p-3 space-y-2">
      <h3 className={`text-lg font-medium text-center ${sideColor}`}>{side}</h3>
      
      {/* Scrollable Fields Container */}
      <div className="h-60 overflow-y-auto overflow-y-scroll" style={{ scrollbarWidth: 'thin' }}>
        <div className="grid grid-cols-2 gap-2 pr-2">
          {fields.map((field) => (
            <div key={field.key} className="space-y-0.5">
              <Label htmlFor={`${side}-${field.key}`} className="text-sm">{field.label}</Label>
              
              {field.type === "checkbox" ? (
                <div className="flex items-center h-8">
                  <Checkbox
                    id={`${side}-${field.key}`}
                    checked={Boolean(stats[field.key as keyof TeamStats] ?? field.defaultValue)}
                    onCheckedChange={(checked) => handleInputChange(field.key, checked)}
                    className="mr-2"
                  />
                  <span className="text-sm text-muted-foreground">
                    {Boolean(stats[field.key as keyof TeamStats] ?? field.defaultValue) ? "Yes" : "No"}
                  </span>
                </div>
              ) : (
                <Input
                  id={`${side}-${field.key}`}
                  type={field.type}
                  placeholder={field.placeholder}
                  value={stats[field.key as keyof TeamStats] ?? field.defaultValue}
                  onChange={(e) => {
                    const value = field.type === "number" ? Number(e.target.value) : e.target.value;
                    handleInputChange(field.key, value);
                  }}
                  className="h-8"
                />
              )}
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}