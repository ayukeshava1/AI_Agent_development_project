import { createContext, useContext, useState } from 'react';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be inside AuthProvider');
  return context;
};

export const AuthProvider = ({ children }) => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);  // Fake initial
  const [token, setToken] = useState(null);  // Stub JWT

  const login = (email, password) => {
    // Stub: Always "success" for now
    setIsLoggedIn(true);
    setToken('fake-jwt-token');
    console.log(`Logged in as ${email}`);  // Real: Axios POST /login
  };

  const logout = () => {
    setIsLoggedIn(false);
    setToken(null);
  };

  return (
    <AuthContext.Provider value={{ isLoggedIn, token, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};