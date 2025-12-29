import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import Home from './pages/Home';
import Converter from './pages/Converter';
import Files from './pages/Files';
import Gallery from './pages/Gallery';
import Resources from './pages/Resources';

function App() {
  return (
    <Router>
      <div className="min-h-screen text-white">
        <Header />
        <div className="flex">
          <Sidebar />
          <main className="ml-64 flex-1 p-8">  {/* ml-64 = sidebar width */}
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/converter" element={<Converter />} />
              <Route path="/files" element={<Files />} />
              <Route path="/gallery" element={<Gallery />} />
              <Route path="/resources" element={<Resources />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;