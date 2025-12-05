import Link from "next/link";
import { Button } from "@/components/ui/button";

export function Navigation() {
  return (
    <div className="absolute top-4 right-4 flex gap-2">
      <Link href="/">
        <Button variant="outline" className="cursor-pointer">
          Home
        </Button>
      </Link>
      <Link href="/draft">
        <Button variant="outline" className="cursor-pointer">
          Draft
        </Button>
      </Link>
      <Link href="/live">
        <Button variant="outline" className="cursor-pointer">
          Live
        </Button>
      </Link>
    </div>
  );
}