import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'SIGMA Signal Analysis | Desktop RF Inspection & Blind Decoding Platform',
  description:
    'SIGMA Signal Analysis is a premium desktop platform for Windows & macOS for RF signal inspection, 18-class hybrid ML modulation classification, blind FEC/interleaver identification, and telemetry decoding.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap"
          rel="stylesheet"
        />
      </head>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
