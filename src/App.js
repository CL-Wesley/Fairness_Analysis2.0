// src/App.js

import React, { useState } from 'react';
import AnalysisForm from './components/AnalysisFormNew';
import ResultsDashboard from './components/ResultsDashboard';
import WelcomeState from './components/WelcomeState';
import LoadingState from './components/LoadingState';
import { Alert } from 'react-bootstrap';
import { Shield, Settings, FileText, HelpCircle, Menu, X } from 'lucide-react';
import './App.css';

function App() {
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const toggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  return (
    <div className="app-container">
      {/* Professional Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="header-left">
            <button
              className="sidebar-toggle"
              onClick={toggleSidebar}
              aria-label="Toggle sidebar"
            >
              {sidebarCollapsed ? <Menu size={20} /> : <X size={20} />}
            </button>
            <div className="brand">
              <Shield className="brand-icon" size={24} />
              <span className="brand-text">AI Fairness Auditor</span>
            </div>
          </div>
          <nav className="header-nav">
            <a href="#" className="nav-item active">
              <Settings size={16} />
              Dashboard
            </a>
            <a href="#" className="nav-item">
              <FileText size={16} />
              Documentation
            </a>
            <a href="#" className="nav-item">
              <HelpCircle size={16} />
              Help
            </a>
          </nav>
        </div>
      </header>

      <div className="app-body">
        {/* Sidebar Configuration Panel */}
        <aside className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
          <div className="sidebar-content">
            <div className="sidebar-header">
              <Settings className="sidebar-icon" size={20} />
              <h3 className="sidebar-title">Analysis Configuration</h3>
            </div>
            <AnalysisForm
              setIsLoading={setIsLoading}
              setResults={setResults}
              setError={setError}
              isLoading={isLoading}
            />
          </div>
        </aside>

        {/* Main Content Area */}
        <main className={`main-content ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
          {error && (
            <Alert
              variant="danger"
              className="error-alert"
              onClose={() => setError('')}
              dismissible
            >
              <strong>Analysis Error:</strong> {error}
            </Alert>
          )}

          {isLoading && <LoadingState />}

          {!isLoading && !results && !error && <WelcomeState />}

          {!isLoading && results && <ResultsDashboard data={results} />}
        </main>
      </div>
    </div>
  );
}

export default App;