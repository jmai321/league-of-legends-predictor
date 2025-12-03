import Link from "next/link";
import { Button } from "@/components/ui/button";

export function Navigation() {
  return (
    <div className="absolute top-4 right-4">
      <Link href="/">
        <Button variant="outline" className="cursor-pointer">
          Home
        </Button>
      </Link>
    </div>
  );
}