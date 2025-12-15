
import React from 'react';
import { useTheme } from '../ThemeContext';

type StatusPanelProps = {
  status: string;
  url: string;
  onClick: () => void;
};

export const StatusPanel = ({ status, url, onClick }: StatusPanelProps) => {
  const { theme } = useTheme();

  const panelStyle: React.CSSProperties = {
    position: 'absolute',
    bottom: 10,
    left: 10,
    padding: '8px 12px',
    backgroundColor: theme === 'dark' ? '#2a2a2a' : '#f0f0f0',
    color: theme === 'dark' ? '#f0f0f0' : '#213547',
    border: `1px solid ${theme === 'dark' ? '#444' : '#ddd'}`,
    borderRadius: 5,
    zIndex: 10,
    cursor: 'pointer',
    fontSize: '0.9em',
  };

  const statusIndicatorStyle: React.CSSProperties = {
    display: 'inline-block',
    width: 10,
    height: 10,
    borderRadius: '50%',
    marginRight: 8,
    backgroundColor: status === 'Connected' ? 'green' : 'red',
  };

  return (
    <div style={panelStyle} onClick={onClick}>
      <span style={statusIndicatorStyle}></span>
      <span>{status}</span>
      <span style={{ marginLeft: 10, color: '#888' }}>{url}</span>
    </div>
  );
};
