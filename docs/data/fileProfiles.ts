export interface FileProfile {
  key: string;
  filename: string;
  badge: string;
  snr: string;
  cfo: string;
  rate: string;
  cls: string;
  evm: string;
  evmPeak: string;
  phaseErr: string;
  lock: string;
  peaks: number[];
  decodedText: string;
  decodedStatus: string;
}

export const FILE_PROFILES: Record<string, FileProfile> = {
  navtex: {
    key: 'navtex',
    filename: 'navtex_518kHz.wav',
    badge: 'SITOR-B',
    snr: '15.2 dB',
    cfo: '+1842.3 Hz',
    rate: '100.00 Bd',
    cls: '2-FSK (Hybrid 99.8%)',
    evm: '3.42 %',
    evmPeak: '6.18 %',
    phaseErr: '0.84°',
    lock: 'LOCKED (100%)',
    peaks: [0.42, 0.58],
    decodedText: `ZCZC EA33 280800 UTC FEB 23\nNITON RADIO (E) - NAVIGATIONAL WARNING\nEDDYSTONE ROCKS LIGHTHOUSE E1 BUOY UNLIT.\nNNNN`,
    decodedStatus: `✔ 10/10 NAVTEX Messages Decoded • BER 0.00%`,
  },
  rs41: {
    key: 'rs41',
    filename: 'rs41_radiosonde.iq',
    badge: 'GFSK',
    snr: '12.4 dB',
    cfo: '-3120.0 Hz',
    rate: '4800.0 Bd',
    cls: 'GFSK (Hybrid 99.9%)',
    evm: '4.15 %',
    evmPeak: '7.92 %',
    phaseErr: '1.12°',
    lock: 'LOCKED (99.9%)',
    peaks: [0.35, 0.65],
    decodedText: `[RS41-SG SONDE TELEMETRY]\nFRAME #1842 • ID: W3420194\nLAT: 52.5200° N • LON: 13.4050° E • ALT: 18,420m\nTEMP: -52.4°C • PRESS: 64.2 hPa • HUM: 12%`,
    decodedStatus: `✔ RS41 Vaisala Telemetry Frame Sync OK • CRC PASS`,
  },
  noaa: {
    key: 'noaa',
    filename: 'noaa18_apt.sigmf',
    badge: 'FM Sub',
    snr: '18.7 dB',
    cfo: '+0.0 Hz',
    rate: '2.00 Hz',
    cls: 'FM / Subcarrier (100%)',
    evm: '1.85 %',
    evmPeak: '3.10 %',
    phaseErr: '0.45°',
    lock: 'LOCKED (100%)',
    peaks: [0.50],
    decodedText: `[NOAA-18 APT SATELLITE IMAGE DECODER]\nCarrier Freq: 137.9125 MHz • Video Subcarrier: 2400 Hz\nLine Rate: 2 Hz (120 LPM) • Sync A: 2080 Hz • Sync B: 1600 Hz\nStatus: 1097 Lines Rasterized to PNG Format`,
    decodedStatus: `✔ NOAA APT Telemetry & Image Frame Extracted`,
  },
  rtty: {
    key: 'rtty',
    filename: 'dwd_rtty_50bd.wav',
    badge: 'Baudot',
    snr: '14.1 dB',
    cfo: '+450.0 Hz',
    rate: '50.00 Bd',
    cls: '2-FSK / Baudot (99.5%)',
    evm: '3.88 %',
    evmPeak: '6.45 %',
    phaseErr: '0.92°',
    lock: 'LOCKED (100%)',
    peaks: [0.45, 0.55],
    decodedText: `ZCZC CQ CQ CQ DE DWD OFFENBACH\nGERMAN WEATHER SERVICE MARINE BULLETIN\nBALTIC SEA: SW 4 TO 5, INCREASING 6 LATER. SHOWERS.\nNNNN`,
    decodedStatus: `✔ DWD RTTY Baudot Code Shift 450Hz Decoded`,
  },
};
