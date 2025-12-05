import { Navigation } from "@/components/Navigation";

interface PageLayoutProps {
  title: string;
  children: React.ReactNode;
}

export function PageLayout({ title, children }: PageLayoutProps) {
  return (
    <div className="min-h-screen bg-background relative">
      <Navigation />
      
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-foreground">
            {title}
          </h1>
        </div>
        
        {children}
      </div>
    </div>
  );
}