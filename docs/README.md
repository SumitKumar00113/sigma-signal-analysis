# SIGMA Signal Analysis — Web Documentation & Landing App

This directory contains the interactive web application and documentation for **SIGMA Signal Analysis**, built with **Next.js 15 (App Router)** and **React 19**.

## Architecture & Project Structure

The project has been refactored into clean, modular components:

```
docs/
├── app/
│   ├── layout.tsx         # Root layout with fonts, metadata, and hydration warning suppression
│   ├── page.tsx           # Main landing page composing all subcomponents
│   └── globals.css        # CSS design system tokens, reset, animations & utility classes
├── components/            # Modular React client components
│   ├── Navbar.tsx         # Sticky top navigation bar & links
│   ├── Hero.tsx           # Hero header with Windows & macOS setup download CTAs
│   ├── AmbientGlows.tsx   # Cinematic background lighting & glow effects
│   ├── BentoGrid.tsx      # All-in-one capability bento grid
│   ├── LargeShowcase.tsx   # Integrated pipeline runner preview
│   ├── CarouselShowcase.tsx # Interactive workbench suite engine carousel (with auto-rotation)
│   ├── DualShowcase.tsx    # Code sample editor showcase
│   ├── FinalCta.tsx        # 3D glowing developer orb & download callout
│   ├── Footer.tsx          # Multi-column footer & copyright
│   ├── bento/
│   │   └── GemGraphic.tsx  # 3D iridescent gem SVG graphic
│   └── workbench/
│       ├── Workbench.tsx   # Workbench frame & active tab state management
│       ├── Sidebar.tsx     # Off-air capture file tree & DSP parameters dock
│       ├── SpectrumCanvas.tsx    # 60 FPS HTML5 Canvas FFT real-time baseband simulator
│       ├── ConstellationCanvas.tsx # Interactive HTML5 Canvas constellation diagram simulator
│       └── DecodeTerminal.tsx    # Decoded telemetry terminal output viewer
├── data/
│   └── fileProfiles.ts    # Dataset profiles for NAVTEX, RS41, NOAA APT & RTTY off-air captures
├── next.config.mjs        # Next.js 15 configuration (static export `output: 'export'`)
├── package.json           # Dependencies and npm scripts
└── tsconfig.json          # TypeScript compiler configuration
```

## Available Scripts

In the `docs` directory, you can run:

### `npm run dev`
Runs the app in development mode at [http://localhost:3000](http://localhost:3000).  
The page will reload when you make edits.

### `npm run build`
Builds the app for production and exports static HTML/JS output to `out/`.  
This generates static files ready for deployment to GitHub Pages or static web hosting.
