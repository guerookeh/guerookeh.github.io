import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [
    Component.Flex({
      components: [{ Component: Component.Darkmode() }, { Component: Component.ReaderMode() }],
    }),
  ],
  afterBody: [],
  footer: Component.Footer({
    links: {
    },
  }),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.ConditionalRender({
      component: Component.Breadcrumbs(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.ArticleTitle(),
    Component.ContentMeta(),
    Component.TagList(),
  ],
  left: [],
  right: [
    Component.ConditionalRender({
      component: Component.Search(),
      condition: (page) => page.fileData.slug == "index",
    }),
    Component.ConditionalRender({
      component: Component.Graph({
        localGraph: {
          depth: -1,
          scale: 0.9,
          focusOnHover: true,
          enableRadial: true,
        },
      }),
      condition: (page) => page.fileData.slug == "index",
    }),
    Component.ConditionalRender({
      component: Component.AllTags(),
      condition: (page) => page.fileData.slug == "index",
    }),
    Component.ConditionalRender({
      component: Component.DesktopOnly(Component.TableOfContents()),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.ConditionalRender({
      component: Component.Backlinks(),
      condition: (page) => page.fileData.slug !== "index",
    }),
  ],
  afterBody: [
    Component.ConditionalRender({
      component: Component.AllNotes(),
      condition: (page) => page.fileData.slug == "index",
    })
  ]
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.Breadcrumbs(), Component.ArticleTitle(), Component.ContentMeta()],
  left: [],
  right: [],
}
