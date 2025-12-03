"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

import { CHAMPIONS } from '@/src/constants/champions';

interface ChampionSelectorProps {
  value?: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

export function ChampionSelector({ value, onChange, placeholder = "Select champion..." }: ChampionSelectorProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const searchInputRef = useRef<HTMLInputElement>(null);

  const filteredChampions = CHAMPIONS.filter(champion =>
    champion.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleOpenChange = (open: boolean) => {
    setIsOpen(open);
    if (open) {
      setSearchTerm("");
      // Focus the input after a short delay to ensure it's rendered
      setTimeout(() => {
        searchInputRef.current?.focus();
      }, 100);
    }
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setSearchTerm(e.target.value);
  };

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    e.stopPropagation();
  };

  const handleInputClick = (e: React.MouseEvent<HTMLInputElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  return (
    <div className="space-y-2">
      <Select value={value} onValueChange={onChange} open={isOpen} onOpenChange={handleOpenChange}>
        <SelectTrigger className="cursor-pointer h-12 justify-center">
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent className="w-full min-w-[250px]">
          <div className="p-2 sticky top-0 bg-background z-10">
            <Input
              ref={searchInputRef}
              placeholder="Search champions..."
              value={searchTerm}
              onChange={handleSearchChange}
              onKeyDown={handleInputKeyDown}
              onClick={handleInputClick}
              className="mb-2"
              autoFocus
            />
          </div>
          <div className="max-h-64 overflow-y-auto">
            {filteredChampions.length > 0 ? (
              filteredChampions.map((champion) => (
                <SelectItem 
                  key={champion} 
                  value={champion}
                  className="cursor-pointer text-center justify-center"
                >
                  {champion}
                </SelectItem>
              ))
            ) : (
              <div className="p-2 text-sm text-muted-foreground text-center">
                No champions found
              </div>
            )}
          </div>
        </SelectContent>
      </Select>
    </div>
  );
}