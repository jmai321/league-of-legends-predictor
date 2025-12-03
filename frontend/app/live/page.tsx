import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Navigation } from "@/src/components/Navigation";

export default function LivePredictionPage() {
  return (
    <div className="min-h-screen bg-background relative">
      <Navigation />
      
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-foreground">
            Live Prediction
          </h1>
        </div>
        
        <Card className="max-w-6xl mx-auto p-6">
          <div className="grid md:grid-cols-2 gap-8">
            {/* Left side - Set Parameters */}
            <div className="space-y-6">
              <h2 className="text-2xl font-semibold text-center">Set Parameters</h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Game Time: 15 min</label>
                  <div className="p-3 border rounded bg-muted/50">
                    {/* Placeholder for time input */}
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Gold Diff: 5k</label>
                  <div className="p-3 border rounded bg-muted/50">
                    {/* Placeholder for gold diff input */}
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Dragons: Mountain</label>
                  <div className="p-3 border rounded bg-muted/50">
                    {/* Placeholder for dragons dropdown */}
                  </div>
                </div>
                
                {Array.from({ length: 2 }, (_, i) => (
                  <div key={i}>
                    <div className="text-muted-foreground">...</div>
                  </div>
                ))}
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