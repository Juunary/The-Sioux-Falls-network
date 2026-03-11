// ============================================================
// NavBar — top navigation bar for multi-page layout
// ============================================================

import React from 'react';
import { NavLink } from 'react-router-dom';

const LINKS = [
  { to: '/simulator',   label: 'Simulator' },
  { to: '/datasets',    label: 'Datasets'  },
  { to: '/training',    label: 'Training'  },
  { to: '/experiments', label: 'Experiments' },
  { to: '/comparison',  label: 'Comparison'  },
] as const;

export default function NavBar() {
  return (
    <nav className="navbar">
      <span className="navbar-brand">Sioux Falls RL</span>
      <div className="navbar-links">
        {LINKS.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              isActive ? 'nav-link nav-link-active' : 'nav-link'
            }
          >
            {label}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
