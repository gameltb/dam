import { Music, Waves } from "lucide-react";
import React from "react";

import { cn } from "@/lib/utils";

interface AudioRendererProps {
  autoPlay?: boolean;
  controls?: boolean;
  url: string;
}

export const AudioRenderer: React.FC<AudioRendererProps> = ({ autoPlay = false, controls = true, url }) => {
  return (
    <div className="relative w-full h-full flex flex-col items-center justify-center p-4 bg-gradient-to-b from-primary/5 to-primary/10 rounded-[inherit] overflow-hidden group">
      {/* Background decoration */}
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-primary/20 to-transparent" />

      {/* Dynamic Waveform Placeholder */}
      <div className="relative flex items-center justify-center w-16 h-16 mb-4">
        <div className="absolute inset-0 bg-primary/20 rounded-full animate-ping opacity-20" />
        <div className="relative z-10 w-12 h-12 rounded-full bg-primary text-primary-foreground flex items-center justify-center shadow-lg shadow-primary/20">
          <Music size={20} />
        </div>

        {/* Decorative Wave lines */}
        <div className="absolute -bottom-2 flex gap-0.5 items-center">
          {[1, 2, 3, 4, 5].map((i) => (
            <div
              className={cn("w-1 bg-primary/40 rounded-full animate-bounce", i % 2 === 0 ? "h-3" : "h-5")}
              key={i}
              style={{ animationDelay: `${i * 0.15}s` }}
            />
          ))}
        </div>
      </div>

      <audio
        autoPlay={autoPlay}
        className="w-full h-8 mt-2 opacity-80 hover:opacity-100 transition-opacity"
        controls={controls}
        src={url}
      />

      <div className="mt-3 flex items-center gap-2 text-[10px] font-bold text-primary/40 uppercase tracking-widest">
        <Waves size={10} />
        <span>Audio Stream</span>
      </div>
    </div>
  );
};
