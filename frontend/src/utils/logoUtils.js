export const getTeamAliases = (teamName) => {
  if (!teamName) return '';
  const name = teamName.toUpperCase().trim();
  const aliases = {
    'CHENNAI SUPER KINGS': 'CSK',
    'DELHI CAPITALS': 'DC',
    'GUJARAT TITANS': 'GT',
    'KOLKATA KNIGHT RIDERS': 'KKR',
    'LUCKNOW SUPER GIANTS': 'LSG',
    'MUMBAI INDIANS': 'MI',
    'PUNJAB KINGS': 'PK',
    'KINGS XI PUNJAB': 'PK',
    'PBKS': 'PK',
    'KXIP': 'PK',
    'ROYAL CHALLENGERS BENGALURU': 'RCB',
    'ROYAL CHALLENGERS BANGALORE': 'RCB',
    'RAJASTHAN ROYALS': 'RR',
    'SUNRISERS HYDERABAD': 'SRH',
    'SH': 'SRH'
  };
  return aliases[name] || name;
};

export const getTeamLogo = (teamName) => {
  const key = getTeamAliases(teamName);
  
  const map = {
    'CSK': 'CSK.jpg',
    'DC': 'DC.jpg',
    'GT': 'GT.jpg',
    'KKR': 'KKR.jpg',
    'LSG': 'LSG.png',
    'MI': 'MI.jpg',
    'PK': 'PK.jpg',
    'RCB': 'RCB.jpg',
    'RR': 'RR.jpg',
    'SRH': 'SRH.jpg',
  };

  const filename = map[key] || `${key}.jpg`;
  return `/logos/${filename}`;
};

export const getTeamTheme = (teamName) => {
  const key = getTeamAliases(teamName);
  
  const themes = {
    'CSK': { primary: '#FACC15', secondary: '#EAB308' }, // Yellow
    'DC': { primary: '#2563EB', secondary: '#1D4ED8' },  // Blue
    'GT': { primary: '#1E3A8A', secondary: '#4F46E5' },  // Navy / Indigo
    'KKR': { primary: '#A855F7', secondary: '#9333EA' }, // Purple
    'LSG': { primary: '#06B6D4', secondary: '#0891B2' }, // Cyan
    'MI': { primary: '#3B82F6', secondary: '#2563EB' },  // Light Blue
    'PK': { primary: '#EF4444', secondary: '#DC2626' },  // Red
    'RCB': { primary: '#EF4444', secondary: '#000000' }, // Red/Black
    'RR': { primary: '#EC4899', secondary: '#BE185D' },  // Pink
    'SRH': { primary: '#F97316', secondary: '#EA580C' }, // Orange
  };

  return themes[key] || { primary: '#94a3b8', secondary: '#475569' }; // default slate
};

export const getTeamLogoFit = (teamName) => {
  const key = getTeamAliases(teamName);
  const maintainOriginalAspect = ['CSK', 'KKR', 'SRH', 'RCB'];
  
  if (maintainOriginalAspect.includes(key)) {
    return 'object-contain p-[3px]';
  }
  return 'object-cover scale-[1.15]';
};
