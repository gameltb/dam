import { type ReactNode, useEffect } from "react";

import { useUiStore } from "@/store/uiStore";
import { Theme } from "@/types";

import { ThemeContext } from "./contexts/ThemeContext";

const ThemeProvider = ({ children }: { children: ReactNode }) => {
  const theme = useUiStore((s) => s.settings.theme);
  const setSettings = useUiStore((s) => s.setSettings);

  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove(Theme.LIGHT, Theme.DARK);
    root.classList.add(theme);
  }, [theme]);

  const toggleTheme = () => {
    setSettings({
      theme: theme === Theme.LIGHT ? Theme.DARK : Theme.LIGHT,
    });
  };

  return <ThemeContext.Provider value={{ theme, toggleTheme }}>{children}</ThemeContext.Provider>;
};

export default ThemeProvider;
