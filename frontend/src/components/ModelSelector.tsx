"use client";

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { PREDICTION_MODELS } from "@/src/constants/gameStats";
import type { PredictionModel } from "@/src/types";

interface ModelSelectorProps {
  value: PredictionModel;
  onChange: (model: PredictionModel) => void;
}

export function ModelSelector({ value, onChange }: ModelSelectorProps) {
  return (
    <div className="space-y-2">
      <Label htmlFor="model-select">Prediction Model</Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger id="model-select" className="w-full">
          <SelectValue placeholder="Select prediction model" />
        </SelectTrigger>
        <SelectContent>
          {PREDICTION_MODELS.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}