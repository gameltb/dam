import React, { useEffect } from "react";
import { useSettingsStore } from "@/store/ui/settingsStore";
import { ThemeContext } from "@/contexts/ThemeContext";
import { Theme } from "@/types";

/**
 * ThemeProvider
 * Manages the application's visual theme and persists it to SettingsStore.
 */
export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { theme, setSettings } = useSettingsStore();

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

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};
