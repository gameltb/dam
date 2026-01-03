import { type ReactNode, useEffect } from "react";
import { ThemeContext } from "./contexts/ThemeContext";
import { useUiStore } from "./store/uiStore";

const ThemeProvider = ({ children }: { children: ReactNode }) => {
  const theme = useUiStore((s) => s.settings.theme);
  const setSettings = useUiStore((s) => s.setSettings);

  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove("light", "dark");
    root.classList.add(theme);
  }, [theme]);

  const toggleTheme = () => {
    setSettings({ theme: theme === "light" ? "dark" : "light" });
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export default ThemeProvider;
