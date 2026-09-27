'use client';

import AmbientGlows from '@/components/AmbientGlows';
import Navbar from '@/components/Navbar';
import Hero from '@/components/Hero';
import Workbench from '@/components/workbench/Workbench';
import BentoGrid from '@/components/BentoGrid';
import LargeShowcase from '@/components/LargeShowcase';
import CarouselShowcase from '@/components/CarouselShowcase';
import DualShowcase from '@/components/DualShowcase';
import FinalCta from '@/components/FinalCta';
import Footer from '@/components/Footer';

export default function Home() {
  return (
    <main>
      <AmbientGlows />
      <Navbar />
      <Hero />
      <Workbench />
      <BentoGrid />
      <LargeShowcase />
      <CarouselShowcase />
      <DualShowcase />
      <FinalCta />
      <Footer />
    </main>
  );
}
