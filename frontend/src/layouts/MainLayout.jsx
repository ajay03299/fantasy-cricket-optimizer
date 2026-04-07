import React from 'react';
import { Outlet, NavLink } from 'react-router-dom';
import { BrainCircuit, LayoutDashboard, User, Trophy, Settings } from 'lucide-react';
import { cn } from '../utils';

export default function MainLayout() {
  const navItems = [
    { label: 'Matches', path: '/', icon: Trophy },
    { label: 'Optimizer', path: '/optimizer', icon: LayoutDashboard },
    { label: 'Profile', path: '/profile', icon: User },
    { label: 'Settings', path: '/settings', icon: Settings },
  ];

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-white font-sans overflow-hidden">
      {/* Top Navigation Bar */}
      <nav className="flex items-center justify-between px-6 py-4 bg-slate-900 border-b border-slate-800 shadow-xl z-20 shrink-0">
        <div className="flex items-center gap-8">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="p-2 bg-emerald-400/10 rounded-lg border border-emerald-400/20 shadow-[0_0_15px_rgba(52,211,153,0.15)]">
              <BrainCircuit className="w-5 h-5 text-emerald-400" />
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent hidden sm:block">FCML Studio</span>
          </div>

          {/* Nav Links */}
          <div className="flex items-center gap-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) => cn(
                    "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300",
                    isActive 
                      ? "bg-slate-800 text-emerald-400 shadow-[inset_0_-2px_0_rgba(52,211,153,1)]"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                  )}
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden md:inline">{item.label}</span>
                </NavLink>
              );
            })}
          </div>
        </div>

        {/* User Profile Summary */}
        <div className="flex items-center gap-4">
          <div className="flex-col items-end hidden sm:flex">
            <span className="text-sm font-bold text-slate-200">Varshith M.</span>
            <span className="text-[10px] text-emerald-400 font-mono tracking-wider uppercase">Pro Tier</span>
          </div>
          <div className="w-10 h-10 rounded-full bg-slate-800 border-2 border-emerald-400/50 flex items-center justify-center overflow-hidden cursor-pointer hover:border-emerald-400 transition-colors">
            <User className="w-5 h-5 text-slate-400" />
          </div>
        </div>
      </nav>

      {/* Main Content Render Area */}
      <main className="flex-1 overflow-hidden relative flex flex-col">
        <Outlet />
      </main>
    </div>
  );
}
