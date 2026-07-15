// cytoscape-cose-bilkent ships no types; it's a cytoscape extension registered via
// cytoscape.use(). The default export is the extension registrar function.
declare module 'cytoscape-cose-bilkent' {
  const ext: (cy: unknown) => void;
  export default ext;
}
