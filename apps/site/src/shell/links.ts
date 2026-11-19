// The addresses of the site pages. The server serves the built site at "/".

export const links = {
    home: '/',
    // The page of one entry, and its developer pages.
    entry: (id: string): string => `/${id}/`,
    tests: (id: string): string => `/${id}/dev/`,
    benchmark: (id: string): string => `/${id}/benchmark/`,
};
