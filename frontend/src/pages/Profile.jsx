import React from 'react';
import { User, Trophy, Star, History, Target, Zap, ChevronRight, Activity } from 'lucide-react';

export default function Profile() {
  const stats = [
    { label: "Optimizations Run", value: "2,408", icon: Activity, color: "text-emerald-400" },
    { label: "Average Accuracy", value: "78.4%", icon: Target, color: "text-cyan-400" },
    { label: "Matches Participated", value: "145", icon: Trophy, color: "text-amber-400" },
    { label: "Pro Credits", value: "12,000", icon: Zap, color: "text-slate-300" }
  ];

  const recentLineups = [
    { date: "Oct 24, 2026", match: "IND vs AUS", score: 845.5, status: "Won" },
    { date: "Oct 20, 2026", match: "RCB vs CSK", score: 712.0, status: "Lost" },
    { date: "Oct 18, 2026", match: "ENG vs SA", score: 920.5, status: "Won" }
  ];

  return (
    <div className="flex-1 p-8 overflow-y-auto w-full max-w-7xl mx-auto custom-scrollbar">
      
      {/* Profile Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-8 mb-8 relative overflow-hidden shadow-xl">
        <div className="absolute top-0 right-0 w-64 h-64 bg-emerald-500/10 rounded-full blur-3xl -mr-20 -mt-20 pointer-events-none"></div>
        
        <div className="flex flex-col md:flex-row items-center gap-8 relative z-10">
          <div className="relative group">
            <div className="w-32 h-32 rounded-full bg-slate-950 border-4 border-emerald-500/50 flex items-center justify-center overflow-hidden shadow-[0_0_30px_rgba(16,185,129,0.2)]">
              <User className="w-16 h-16 text-slate-500" />
            </div>
            <div className="absolute inset-0 bg-slate-950/80 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer">
               <span className="text-sm font-semibold text-emerald-400">Edit Photo</span>
            </div>
          </div>
          
          <div className="flex-1 text-center md:text-left">
            <h1 className="text-3xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent mb-2">Varshith M.</h1>
            <p className="text-slate-400 text-sm mb-4">Fantasy Cricket Machine Learning Enthusiast</p>
            
            <div className="flex flex-wrap justify-center md:justify-start gap-3">
              <span className="flex items-center gap-1.5 px-3 py-1 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-full text-xs font-bold uppercase tracking-wider">
                <Star className="w-3 h-3" /> Pro Tier
              </span>
              <span className="flex items-center gap-1.5 px-3 py-1 bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 rounded-full text-xs font-bold uppercase tracking-wider">
                Top 5% Model
              </span>
            </div>
          </div>
          
          <div className="bg-slate-950 rounded-2xl p-6 border border-slate-800/80 shadow-inner min-w-[200px] text-center">
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Current Rank</h3>
            <div className="text-4xl font-black text-white drop-shadow-[0_0_15px_rgba(255,255,255,0.2)]">#4,208</div>
            <div className="text-[10px] text-emerald-400 font-medium mt-1">↑ 145 positions this week</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Statistics Grid */}
        <div className="lg:col-span-2 space-y-8">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
             {stats.map((stat, i) => {
               const Icon = stat.icon;
               return (
                 <div key={i} className="bg-slate-900 border border-slate-800 rounded-2xl p-6 group hover:border-slate-700 transition-colors">
                   <div className="flex items-center gap-3 mb-4">
                     <div className={`p-2 rounded-lg bg-slate-950 border border-slate-800 shadow-inner ${stat.color} bg-opacity-10`}>
                       <Icon className={`w-5 h-5 ${stat.color}`} />
                     </div>
                     <span className="text-sm font-semibold text-slate-400">{stat.label}</span>
                   </div>
                   <div className="text-3xl font-bold text-white group-hover:scale-[1.02] transition-transform origin-left">{stat.value}</div>
                 </div>
               )
             })}
          </div>
          
          {/* History Chart Placeholder */}
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 flex flex-col h-64 justify-center items-center text-slate-500 relative overflow-hidden group">
             <Activity className="w-12 h-12 mb-4 opacity-50 group-hover:scale-110 transition-transform duration-500 text-cyan-400/50" />
             <p className="font-medium">Performance Metrics Chart locked for Pro Tier</p>
             <button className="mt-4 px-6 py-2 bg-slate-950 border border-slate-800 hover:border-cyan-500 hover:text-cyan-400 rounded-lg text-sm font-semibold transition-all">Unlock Detailed Analytics</button>
             
             {/* Fake graph lines in background */}
             <div className="absolute bottom-0 w-full h-1/2 opacity-10 pointer-events-none" style={{ backgroundImage: 'linear-gradient(to top, rgba(34,211,238,0.2) 0%, transparent 100%)' }}></div>
          </div>
        </div>

        {/* Sidebar / Recent Lineups */}
        <div className="space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl relative h-full">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2 mb-6">
              <History className="w-4 h-4 text-emerald-400" />
              Recent Lineups
            </h3>
            
            <div className="space-y-4">
              {recentLineups.map((lineup, i) => (
                <div key={i} className="bg-slate-950 border border-slate-800 p-4 rounded-xl hover:border-slate-700 transition-colors cursor-pointer group">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">{lineup.date}</span>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-wider ${lineup.status === 'Won' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-red-500/10 text-red-400 border border-red-500/20'}`}>
                      {lineup.status}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="font-semibold text-slate-200">{lineup.match}</span>
                    <span className="font-mono text-cyan-400">{lineup.score.toFixed(1)}</span>
                  </div>
                  <div className="mt-3 flex items-center gap-1 text-[10px] font-bold text-slate-500 group-hover:text-emerald-400 transition-colors uppercase tracking-wider">
                     Load Matrix <ChevronRight className="w-3 h-3" />
                  </div>
                </div>
              ))}
            </div>
            
            <button className="w-full mt-6 py-3 border border-slate-800 hover:bg-slate-800 text-sm font-semibold text-slate-300 rounded-xl transition-colors">
              View Complete History
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}
