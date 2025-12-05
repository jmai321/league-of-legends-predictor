# Frontend

## Setup
```bash
npm install
```

## How to Run
```bash
npm run dev
```

## File Structure

```
frontend/
├── app/                    # Next.js app directory
│   ├── draft/
│   │   └── page.tsx        # Champion draft prediction page
│   ├── live/
│   │   └── page.tsx        # Live game prediction page
│   ├── globals.css         # Global styles
│   ├── layout.tsx          # Root layout
│   └── page.tsx            # Home page
├── api/
│   ├── client.ts           # API client base
│   ├── draft.ts            # Draft prediction API
│   ├── live.ts             # Live prediction API
│   └── index.ts            # API exports
├── components/
│   ├── ui/                 # Shadcn UI components
│   ├── layout/
│   │   └── PageLayout.tsx  # Common page layout
│   ├── ChampionSelector.tsx
│   ├── ModelSelector.tsx
│   └── TeamStatsForm.tsx
├── constants/
│   ├── champions.ts        # Champion data
│   └── gameStats.ts        # Game statistics config
├── lib/
│   ├── transformers.ts     # Data transformers
│   └── utils.ts            # Utility functions
├── types/
│   └── index.ts            # TypeScript definitions
├── components.json         # Shadcn config
├── next.config.js          # Next.js config
├── package.json            # Dependencies
├── tailwind.config.ts      # Tailwind config
└── tsconfig.json           # TypeScript config
```

