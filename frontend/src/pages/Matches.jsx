import React, { useState, useEffect } from 'react';
import { Target, Search, Calendar, ChevronRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { getTeamLogo, getTeamTheme, getTeamLogoFit } from '../utils/logoUtils';

export default function Matches() {
  const [upcomingMatches, setUpcomingMatches] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    fetch('http://localhost:8000/matches')
      .then(res => res.json())
      .then(data => {
        const formattedData = data.map((match) => ({
          ...match,
          status: match.status ? (match.status.charAt(0).toUpperCase() + match.status.slice(1)) : 'Completed',
          matchType: 'T20'
        }));
        setUpcomingMatches(formattedData);
      })
      .catch(console.error);
  }, []);

  const filteredMatches = upcomingMatches.filter(match => 
    match.team1.toLowerCase().includes(searchQuery.toLowerCase()) || 
    match.team2.toLowerCase().includes(searchQuery.toLowerCase()) ||
    match.venue.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleOptimizeClick = (match) => {
    navigate(`/optimizer?matchId=${match.match_id}&t1=${match.team1}&t2=${match.team2}`);
  };

  return (
    <div className="flex-1 p-8 overflow-y-auto w-full max-w-7xl mx-auto custom-scrollbar">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-3">
            <Target className="w-8 h-8 text-emerald-400" />
            Select Match
          </h1>
          <p className="text-slate-400 mt-1">Choose a fixture to begin ML Fantasy Optimization</p>
        </div>
        
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="text"
            placeholder="Search teams or venues..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="bg-slate-900 border border-slate-700 text-white rounded-lg pl-10 pr-4 py-2 text-sm outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 transition-all w-64"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredMatches.map((match) => (
          <div 
            key={match.match_id} 
            onClick={() => handleOptimizeClick(match)}
            className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 hover:bg-slate-800/80 hover:border-emerald-500/50 transition-all duration-300 group cursor-pointer shadow-lg hover:shadow-emerald-900/20 relative overflow-hidden"
          >
            <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/5 rounded-full blur-3xl -mr-10 -mt-10 pointer-events-none group-hover:bg-emerald-500/10 transition-colors"></div>
            
            <div className="flex justify-between items-start mb-6 relative">
              <span className="text-xs font-bold px-3 py-1 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-full">
                {match.status}
              </span>
              <span className="text-xs font-medium text-slate-400 bg-slate-950 px-2 py-1 rounded">
                {match.matchType}
              </span>
            </div>

            <div className="flex items-center justify-between mb-8 relative px-4">
              <div className="flex flex-col items-center">
                <div 
                  className="w-16 h-16 sm:w-20 sm:h-20 rounded-full border flex items-center justify-center transition-all overflow-hidden bg-white relative shadow-md"
                  style={{ borderColor: getTeamTheme(match.team1).primary }}
                >
                  <img src={getTeamLogo(match.team1)} alt={match.team1} className={`w-full h-full ${getTeamLogoFit(match.team1)}`} />
                </div>
                <span className="mt-3 text-sm font-bold tracking-wide" style={{ color: getTeamTheme(match.team1).secondary }}>{match.team1}</span>
              </div>
              
              <div className="flex flex-col items-center px-4">
                <span className="text-[10px] font-black text-slate-500 mb-1 tracking-[0.2em] bg-slate-900/80 px-2 py-1 rounded-md border border-slate-800/50 uppercase shadow-inner">VS</span>
                <div className="w-16 h-[1px] bg-gradient-to-r from-transparent via-slate-700 to-transparent"></div>
              </div>

              <div className="flex flex-col items-center">
                 <div 
                  className="w-16 h-16 sm:w-20 sm:h-20 rounded-full border flex items-center justify-center transition-all overflow-hidden bg-white relative shadow-md"
                  style={{ borderColor: getTeamTheme(match.team2).primary }}
                >
                  <img src={getTeamLogo(match.team2)} alt={match.team2} className={`w-full h-full ${getTeamLogoFit(match.team2)}`} />
                </div>
                <span className="mt-3 text-sm font-bold tracking-wide" style={{ color: getTeamTheme(match.team2).secondary }}>{match.team2}</span>
              </div>
            </div>

            <div className="bg-slate-950 rounded-xl p-4 border border-slate-800/50 flex flex-col gap-2 relative">
              <div className="flex items-center gap-2 text-sm text-slate-300">
                <Calendar className="w-4 h-4 text-cyan-400" />
                {match.date}
              </div>
              <div className="flex items-center gap-2 text-sm text-slate-300">
                <Target className="w-4 h-4 text-emerald-400" />
                {match.venue}
              </div>
            </div>

            <div className="mt-6 flex items-center justify-between text-sm font-semibold text-slate-400 group-hover:text-emerald-400 transition-colors relative">
              <span>Optimize Lineup</span>
              <ChevronRight className="w-5 h-5 opacity-0 -ml-4 group-hover:opacity-100 group-hover:ml-0 transition-all duration-300" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
