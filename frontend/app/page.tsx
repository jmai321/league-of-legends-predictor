import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

export default function Home() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <h1 className="text-4xl font-bold text-foreground mb-4">
            League of Legends Competitive Predictor
          </h1>
        </div>
        
        <Card className="max-w-4xl mx-auto p-8">
          <div className="grid md:grid-cols-2 gap-8">
            <Link href="/draft" className="block">
              <Button 
                variant="outline" 
                className="w-full h-48 text-xl font-semibold hover:bg-accent cursor-pointer"
              >
                Draft Prediction
              </Button>
            </Link>
            
            <Link href="/live" className="block">
              <Button 
                variant="outline" 
                className="w-full h-48 text-xl font-semibold hover:bg-accent cursor-pointer"
              >
                Live Prediction
              </Button>
            </Link>
          </div>
        </Card>
      </div>
    </div>
  );
}
