# Frontend

Next.js web application for League of Legends match prediction.

## How to Run

```bash
npm install
npm run dev
```

Visit http://localhost:3000 to access the application.

## Available Scripts

```bash
npm run dev      # Start development server
npm run build    # Build for production
npm run start    # Start production server
npm run lint     # Run ESLint
```

## File Structure

```
frontend/
├── Dockerfile                     # Docker container configuration
├── README.md
├── package.json                   # Dependencies and scripts
├── package-lock.json
├── next.config.ts                 # Next.js configuration
├── tsconfig.json                  # TypeScript configuration
├── eslint.config.mjs              # ESLint configuration
├── postcss.config.mjs             # PostCSS configuration
├── components.json                # shadcn/ui component configuration
├── next-env.d.ts                  # Next.js type definitions
│
├── app/                          # Next.js app router pages
│   ├── draft/
│   │   └── page.tsx              # Champion draft prediction interface
│   ├── live/
│   │   └── page.tsx              # Live game prediction interface
│   ├── layout.tsx                # Root layout component
│   ├── page.tsx                  # Home page
│   ├── globals.css               # Global styles and Tailwind imports
│   └── favicon.ico
│
├── components/                   # React components
│   ├── ui/                      # shadcn/ui base components
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   ├── checkbox.tsx
│   │   ├── input.tsx
│   │   ├── label.tsx
│   │   ├── select.tsx
│   │   └── tabs.tsx
│   ├── layout/
│   │   └── PageLayout.tsx        # Common page layout wrapper
│   ├── ChampionSelector.tsx      # Searchable champion dropdown
│   ├── ModelSelector.tsx         # Model selection component
│   ├── Navigation.tsx            # Navigation menu
│   └── TeamStatsForm.tsx         # Live game statistics input form
│
├── api/                          # Frontend API client
│   ├── client.ts                # Base HTTP client configuration
│   ├── draft.ts                 # Draft prediction API calls
│   ├── live.ts                  # Live prediction API calls
│   └── index.ts                 # API exports
│
├── constants/                   # Static data and configuration
│   ├── champions.ts             # Champion names and role mappings
│   └── gameStats.ts             # Default game statistics and field definitions
│
├── lib/                         # Utility functions
│   ├── transformers.ts          # Data transformation utilities
│   └── utils.ts                 # General utility functions
│
└── types/                       # TypeScript type definitions
    └── index.ts                 # Application-wide type definitions
```

## Third-Party Dependencies

**Core Framework:**
- Next.js 16.0.7 - React framework with app router
- React 19.2.0 - UI library
- TypeScript - Type-safe JavaScript

**Styling:**
- Tailwind CSS 4 - Utility-first CSS framework
- @tailwindcss/postcss - PostCSS integration
- tailwind-merge - Merge Tailwind classes
- clsx - Conditional className utility

**UI Components:**
- @radix-ui/react-* - Headless UI primitives
  - react-checkbox - Checkbox component
  - react-label - Label component  
  - react-select - Select dropdown component
  - react-slot - Slot component for composition
  - react-tabs - Tabs component
- class-variance-authority - Component variant management
- lucide-react - Icon library

**Development:**
- ESLint 9 - Code linting
- eslint-config-next - Next.js ESLint rules
- @types/* - TypeScript type definitions

