import { User, Bell } from 'lucide-react';  // Icons
import ThemeToggle from './ThemeToggle.jsx';

const Header = () => (
  <header className="bg-purple-600 p-4 flex justify-between items-center shadow-lg">
    <h1 className="text-2xl font-bold text-white">AI Converter</h1>
    <div className="flex items-center space-x-4">
      <ThemeToggle />
      <Bell className="w-5 h-5 text-white hover:text-orange-300 cursor-pointer" />
      <User className="w-5 h-5 text-white hover:text-orange-300 cursor-pointer" />
      <button className="bg-orange-500 hover:bg-orange-400 text-white px-4 py-2 rounded-md transition-colors">
        Sign In
      </button>
    </div>
  </header>
);

export default Header;