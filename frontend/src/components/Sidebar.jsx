import React from 'react';
import { NavLink } from 'react-router-dom';
import { FaTachometerAlt, FaMapMarkedAlt } from 'react-icons/fa';

const Sidebar = () => {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h2>DisasterDL</h2>
      </div>
      <nav>
        <ul>
          <li>
            <NavLink to="/" className={({ isActive }) => (isActive ? 'active' : '')}>
              <FaTachometerAlt />
              <span>Dashboard</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/analysis" className={({ isActive }) => (isActive ? 'active' : '')}>
              <FaMapMarkedAlt />
              <span>Disaster Analysis</span>
            </NavLink>
          </li>
        </ul>
      </nav>
    </div>
  );
};

export default Sidebar;
