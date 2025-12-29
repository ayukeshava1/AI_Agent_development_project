import { useState, useEffect } from 'react';
import { Moon, Sun } from 'lucide-react';

const ThemeToggle = () => {
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    // Load saved theme from localStorage
    const saved = localStorage.getItem('darkMode') === 'true';
    setDarkMode(saved);
    document.documentElement.classList.toggle('dark', saved);
    console.log('Theme loaded:', saved ? 'Dark' : 'Light');
  }, []);

  const toggle = () => {
    const newMode = !darkMode;
    setDarkMode(newMode);
    localStorage.setItem('darkMode', newMode);
    document.documentElement.classList.toggle('dark', newMode);
    console.log('Theme toggled to:', newMode ? 'Dark' : 'Light');
  };

  return (
    <div className={`bg-white ${darkMode ? 'dark-mode-bg' : ''} p-2 rounded-md`}>
      <button
        onClick={toggle}
        className="p-2 rounded-full bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
        title="Toggle Theme"
      >
        {darkMode ? (
          <Sun className="w-5 h-5 text-yellow-500" />
        ) : (
          <Moon className="w-5 h-5 text-gray-800" />
        )}
      </button>
    </div>
  );
};

export default ThemeToggle;
